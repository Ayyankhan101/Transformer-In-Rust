use candle_core::Result;

use crate::generation::sampling::sample;
use crate::glm::attention_mask::build_glm_mask;
use crate::glm::model::GLMModel;

pub struct GLMGenerator {
    model: GLMModel,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
    max_new_tokens: usize,
    eos_token_id: Option<u32>,
    mask_token_id: u32,
}

impl GLMGenerator {
    pub fn new(
        model: GLMModel,
        mask_token_id: u32,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        repetition_penalty: f64,
        max_new_tokens: usize,
    ) -> Self {
        Self {
            model,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_new_tokens,
            eos_token_id: None,
            mask_token_id,
        }
    }

    #[allow(dead_code)]
    pub fn set_eos_token_id(&mut self, token_id: u32) {
        self.eos_token_id = Some(token_id);
    }

    /// Causal (GPT-style) autoregressive generation.
    /// No blanks — generates new tokens after the prompt.
    pub fn generate(&self, prompt_token_ids: &[u32]) -> Result<Vec<u32>> {
        let device = self.model.embedding.weight.device();
        let mut generated = prompt_token_ids.to_vec();

        for _step in 0..self.max_new_tokens {
            let seq_len = generated.len();
            let mask =
                crate::layers::attention::causal_mask(seq_len, device, candle_core::DType::F32)?;

            let logits = self.model.forward(&generated, 0, &[], &mask)?;

            let last_logits = logits.get(0)?.get(seq_len - 1)?;
            let token_id = sample(
                &last_logits,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                &generated,
                42,
            )?;

            if Some(token_id) == self.eos_token_id {
                break;
            }

            generated.push(token_id);
        }

        Ok(generated)
    }

    /// Blank-infilling generation.
    ///
    /// Given a context prefix and blank lengths, autoregressively fills each
    /// blank position left-to-right within each blank, blanks left-to-right.
    /// The blank-infilling attention mask lets each position attend to:
    /// - all context tokens (bidirectional)
    /// - all earlier blanks (full visibility)
    /// - earlier positions within the same blank (causal within-span)
    pub fn fill_blanks(&self, context: &[u32], blank_lens: &[usize]) -> Result<Vec<u32>> {
        let device = self.model.embedding.weight.device();
        let total_blank_tokens: usize = blank_lens.iter().sum();

        let mut all_tokens = context.to_vec();
        all_tokens.extend(std::iter::repeat(self.mask_token_id).take(total_blank_tokens));

        let mask = build_glm_mask(context.len(), blank_lens, device)?;

        let mut blank_start = context.len();
        for &blank_len in blank_lens.iter() {
            for pos_in_blank in 0..blank_len {
                let position = blank_start + pos_in_blank;

                let logits = self
                    .model
                    .forward(&all_tokens, context.len(), blank_lens, &mask)?;

                let pos_logits = logits.get(0)?.get(position)?;
                let token_id = sample(
                    &pos_logits,
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.repetition_penalty,
                    &all_tokens,
                    42,
                )?;

                all_tokens[position] = token_id;
            }
            blank_start += blank_len;
        }

        Ok(all_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::config::GLMConfig;
    use candle_core::Device;

    fn small_config() -> GLMConfig {
        GLMConfig {
            vocab_size: 1000,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            ffn_dim: 128,
            max_seq_len: 32,
            ..Default::default()
        }
    }

    #[test]
    fn test_glm_generator_new() {
        let device = Device::Cpu;
        let config = small_config();
        let model = GLMModel::new(config.clone(), &device).unwrap();
        let gen = GLMGenerator::new(model, 999, 0.8, 40, 0.9, 1.1, 10);
        assert_eq!(gen.max_new_tokens, 10);
        assert_eq!(gen.mask_token_id, 999);
        assert_eq!(gen.temperature, 0.8);
    }

    #[test]
    fn test_glm_generator_generate_causal() {
        let device = Device::Cpu;
        let config = small_config();
        let model = GLMModel::new(config, &device).unwrap();
        let gen = GLMGenerator::new(model, 999, 0.0, 1, 1.0, 1.0, 5);

        let prompt = vec![10, 20, 30];
        let result = gen.generate(&prompt).unwrap();

        // Should have prompt + generated tokens
        assert!(result.len() >= prompt.len());
        assert!(result.len() <= prompt.len() + 5);
    }

    #[test]
    fn test_glm_generator_fill_blanks() {
        let device = Device::Cpu;
        let config = small_config();
        let model = GLMModel::new(config, &device).unwrap();
        let gen = GLMGenerator::new(model, 999, 0.0, 1, 1.0, 1.0, 5);

        let context = vec![10, 20, 30, 40];
        let blank_lens = vec![2, 1];
        let result = gen.fill_blanks(&context, &blank_lens).unwrap();

        // Should have context + blank tokens
        assert_eq!(result.len(), 4 + 3); // 4 context + 2 + 1 blank
    }
}
