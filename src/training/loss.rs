//! Cross-entropy loss for GLM training.

use candle_core::{Result, Tensor};

/// Mean cross-entropy over the positions carrying a label.
///
/// `logits` is `[1, seq_len, vocab_size]`; `labels[i]` is the target token for
/// position `i`, or `-1` for positions that should not contribute. Returns a
/// scalar with no labelled positions, in which case there is nothing to learn
/// from and the caller should skip the optimizer step.
///
/// This is built from tensor operations on purpose. The previous version summed
/// `ce.to_scalar::<f32>()` into an `f32` and returned `Tensor::new(total, device)`
/// — a fresh leaf with no autograd history — so `loss.backward()` produced an
/// empty gradient store and every training step was a no-op.
pub fn cross_entropy_loss(logits: &Tensor, labels: &[i64]) -> Result<Tensor> {
    let device = logits.device();
    let seq_len = logits.dim(1)?;

    let positions: Vec<u32> = labels
        .iter()
        .take(seq_len)
        .enumerate()
        .filter(|(_, &label)| label >= 0)
        .map(|(i, _)| i as u32)
        .collect();
    let targets: Vec<u32> = labels
        .iter()
        .take(seq_len)
        .filter(|&&label| label >= 0)
        .map(|&label| label as u32)
        .collect();

    if positions.is_empty() {
        return Tensor::new(0.0f32, device);
    }

    let n = positions.len();
    let index = Tensor::from_vec(positions, n, device)?;
    let selected = logits.squeeze(0)?.index_select(&index, 0)?;
    let targets = Tensor::from_vec(targets, n, device)?;

    candle_nn::loss::cross_entropy(&selected.to_dtype(candle_core::DType::F32)?, &targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    use crate::glm::config::GLMConfig;
    use crate::glm::trainable::TrainableGLMModel;

    fn tiny_config() -> GLMConfig {
        GLMConfig {
            vocab_size: 64,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            ffn_dim: 64,
            max_seq_len: 16,
            ..Default::default()
        }
    }

    /// A masked-denoising batch in the shape the trainer produces: some positions
    /// replaced by the mask token, labels holding the originals.
    fn corrupted_batch(config: &GLMConfig) -> (Vec<u32>, Vec<i64>) {
        let tokens: Vec<u32> = vec![3, 9, 14, 2, 41, 7, 33, 5];
        let mask_id = config.vocab_size as u32 - 1;
        let mut inputs = tokens.clone();
        let mut labels = vec![-1i64; tokens.len()];
        for i in [2usize, 5, 7] {
            labels[i] = tokens[i] as i64;
            inputs[i] = mask_id;
        }
        (inputs, labels)
    }

    #[test]
    fn no_labelled_positions_is_zero() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0f32, (1, 4, 10), &device)?;
        let loss = cross_entropy_loss(&logits, &[-1, -1, -1, -1])?;
        assert_eq!(loss.to_scalar::<f32>()?, 0.0);
        Ok(())
    }

    #[test]
    fn matches_hand_computed_log_softmax() -> Result<()> {
        let device = Device::Cpu;
        // One position, one clear winner: loss is -log_softmax(logits)[label].
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 1, 3), &device)?;
        let loss = cross_entropy_loss(&logits, &[2])?.to_scalar::<f32>()?;

        let denom: f32 = [1.0f32, 2.0, 3.0].iter().map(|v| (v - 3.0f32).exp()).sum();
        let expected = -((3.0f32 - 3.0) - denom.ln());
        assert!(
            (loss - expected).abs() < 1e-5,
            "loss {loss} vs expected {expected}"
        );
        Ok(())
    }

    /// The direct assertion for the detached-loss bug: backprop must reach the
    /// model's parameters. On the previous implementation the gradient store came
    /// back empty.
    #[test]
    fn gradients_reach_the_parameters() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let model = TrainableGLMModel::new(config.clone(), &device)?;
        let (inputs, labels) = corrupted_batch(&config);

        let logits = model.forward_causal(&inputs)?;
        let loss = cross_entropy_loss(&logits, &labels)?;
        let grads = loss.backward()?;

        let params = model.param_vars();
        let with_grad = params.iter().filter(|v| grads.get(v).is_some()).count();
        assert!(
            with_grad >= params.len() - 1,
            "only {with_grad} of {} parameters received gradients",
            params.len()
        );
        Ok(())
    }

    /// End-to-end proof that optimisation works: a two-layer model memorises one
    /// batch. Fails flat if gradients stop flowing for any reason.
    #[test]
    fn training_reduces_loss() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let model = TrainableGLMModel::new(config.clone(), &device)?;
        let (inputs, labels) = corrupted_batch(&config);

        let mut optimizer = AdamW::new(
            model.param_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )?;

        let first =
            cross_entropy_loss(&model.forward_causal(&inputs)?, &labels)?.to_scalar::<f32>()?;

        let mut last = first;
        for _ in 0..60 {
            let loss = cross_entropy_loss(&model.forward_causal(&inputs)?, &labels)?;
            last = loss.to_scalar::<f32>()?;
            optimizer.backward_step(&loss)?;
        }

        assert!(
            last < first * 0.3,
            "loss went {first} -> {last}; the model is not learning"
        );
        Ok(())
    }

    /// Parameters must actually change value, not merely receive gradients.
    #[test]
    fn optimizer_updates_parameters() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let model = TrainableGLMModel::new(config.clone(), &device)?;
        let (inputs, labels) = corrupted_batch(&config);

        let before = model.lm_head.as_tensor().copy()?;
        let mut optimizer = AdamW::new(
            model.param_vars(),
            ParamsAdamW {
                lr: 0.05,
                ..Default::default()
            },
        )?;
        let loss = cross_entropy_loss(&model.forward_causal(&inputs)?, &labels)?;
        optimizer.backward_step(&loss)?;

        let delta = (model.lm_head.as_tensor() - &before)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(
            delta > 0.0,
            "lm_head did not change after an optimizer step"
        );
        Ok(())
    }
}
