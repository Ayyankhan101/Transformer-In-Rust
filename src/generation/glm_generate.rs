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
