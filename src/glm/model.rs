use candle_core::{Device, Result, Tensor};

use crate::layers::embedding::Embedding;
use crate::layers::ffn::Activation;
use crate::layers::norm::RMSNorm;
use crate::layers::block::TransformerBlock;

use super::config::GLMConfig;
use super::positions::GLMPositionEncoding;

pub struct GLMModel {
    pub embedding: Embedding,
    pub position_encoding: GLMPositionEncoding,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub lm_head: Tensor,
    _config: GLMConfig,
}

impl GLMModel {
    pub fn new(config: GLMConfig, device: &Device) -> Result<Self> {
        let embedding = Embedding::new(config.vocab_size, config.hidden_dim, device)?;
        let position_encoding = GLMPositionEncoding::new(
            config.max_seq_len,
            config.hidden_dim,
            device,
        )?;

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.hidden_dim,
                config.num_heads,
                config.ffn_dim,
                crate::layers::block::NormType::RMSNorm,
                Activation::SwiGLU,
                device,
            )?);
        }

        let final_norm = RMSNorm::new(config.hidden_dim, config.eps, device)?;
        let lm_head = Tensor::randn(0.0f32, 0.02f32, (config.hidden_dim, config.vocab_size), device)?;

        Ok(Self {
            embedding,
            position_encoding,
            blocks,
            final_norm,
            lm_head,
            _config: config,
        })
    }

    fn apply_positions(&self, x: &Tensor, context_len: usize, blank_lens: &[usize]) -> Result<Tensor> {
        let seq_len = x.dims()[1];
        if context_len == 0 && blank_lens.is_empty() {
            // Causal mode: use simple learned positions
            let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
            let pos_tensor = Tensor::new(pos_ids.as_slice(), x.device())?;
            let pos_emb = self.position_encoding.pos_1_embedding.index_select(&pos_tensor, 0)?;
            x.broadcast_add(&pos_emb.unsqueeze(0)?)
        } else {
            let pos = self.position_encoding.forward(context_len, blank_lens, x.device())?;
            x.broadcast_add(&pos)
        }
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        context_len: usize,
        blank_lens: &[usize],
        mask: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(token_ids)?;
        x = self.apply_positions(&x, context_len, blank_lens)?;

        for block in &self.blocks {
            x = block.forward(&x, Some(mask))?;
        }

        let x = self.final_norm.forward(&x)?;
        x.broadcast_matmul(&self.lm_head.unsqueeze(0)?)
    }

    #[allow(dead_code)]
    pub fn forward_train(
        &self,
        token_ids: &[u32],
        context_len: usize,
        blank_lens: &[usize],
        mask: &Tensor,
        labels: &[i64],
    ) -> Result<Tensor> {
        let logits = self.forward(token_ids, context_len, blank_lens, mask)?;

        // Compute cross-entropy on non -1 labels
        let seq_len = token_ids.len();
        let mut total_loss = 0.0f32;
        let mut count = 0;

        for i in 0..seq_len {
            if labels[i] >= 0 {
                let logits_i = logits.get(0)?.get(i)?;
                let ce = candle_nn::ops::log_softmax(&logits_i, 0)?
                    .get(labels[i] as usize)?
                    .neg()?;
                total_loss += ce.to_scalar::<f32>()?;
                count += 1;
            }
        }

        if count > 0 {
            total_loss /= count as f32;
        }

        Ok(Tensor::new(total_loss, logits.device())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::attention_mask::build_glm_mask;

    #[test]
    fn test_glm_causal_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = GLMConfig::default();
        let model = GLMModel::new(config, &device)?;
        let ids = &[1u32, 2, 3, 4];
        // Causal mode: context_len = len, blank_lens empty
        let mask = build_glm_mask(ids.len(), &[], &device)?;
        let logits = model.forward(ids, ids.len(), &[], &mask)?;
        assert_eq!(logits.dims(), &[1, 4, 51200]);
        Ok(())
    }

    #[test]
    fn test_glm_blank_infill_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = GLMConfig::default();
        let model = GLMModel::new(config, &device)?;
        // total = context_len(3) + blank_lens([2]) = 5 tokens
        let ids = &[1u32, 2, 3, 0, 0];
        let mask = build_glm_mask(3, &[2], &device)?;
        let logits = model.forward(ids, 3, &[2], &mask)?;
        assert_eq!(logits.dims(), &[1, 5, 51200]);
        Ok(())
    }

    #[test]
    fn test_glm_train_loss() -> Result<()> {
        let device = Device::Cpu;
        let config = GLMConfig::default();
        let model = GLMModel::new(config, &device)?;
        let ids = &[1u32, 2, 3, 4];
        let mask = build_glm_mask(ids.len(), &[], &device)?;
        let labels = &[-1i64, -1, 1, 2];
        let loss = model.forward_train(ids, ids.len(), &[], &mask, labels)?;
        assert!(loss.to_scalar::<f32>()?.is_finite());
        Ok(())
    }
}
