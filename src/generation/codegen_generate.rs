use candle_core::Result;

use crate::codegen::kv_cache::KVCache;
use crate::codegen::model::CodeGenModel;
use crate::generation::sampling::sample;

const EOS_TOKEN_ID: u32 = 50256;

pub struct CodeGenGenerator {
    model: CodeGenModel,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
    max_new_tokens: usize,
}

impl CodeGenGenerator {
    pub fn new(
        model: CodeGenModel,
        temperature: f64,
        top_k: usize,
        top_p: f64,
        repetition_penalty: f64,
        max_new_tokens: usize,
    ) -> Self {
        Self { model, temperature, top_k, top_p, repetition_penalty, max_new_tokens }
    }

    pub fn generate(&self, prompt_token_ids: &[u32]) -> Result<Vec<u32>> {
        let gen_start = std::time::Instant::now();

        let mut cache: Option<Vec<KVCache>> = None;
        let positions: Vec<usize> = (0..prompt_token_ids.len()).collect();
        let logits = self.model.forward_with_cache(prompt_token_ids, &positions, &mut cache)?;

        let last_logits = logits.get(0)?.get(prompt_token_ids.len() - 1)?;
        let token_id = sample(&last_logits, self.temperature, self.top_k, self.top_p, self.repetition_penalty, prompt_token_ids, 42)?;

        let mut generated = prompt_token_ids.to_vec();
        generated.push(token_id);

        eprintln!("  [prefill done in {:.1}s]", gen_start.elapsed().as_secs_f64());

        for step in 1..self.max_new_tokens {
            let step_start = std::time::Instant::now();
            let input_id = [generated.last().copied().unwrap()];
            let pos = vec![generated.len() - 1];

            let logits = self.model.forward_with_cache(&input_id, &pos, &mut cache)?;
            let token_logits = logits.get(0)?.get(0)?;

            let token_id = sample(&token_logits, self.temperature, self.top_k, self.top_p, self.repetition_penalty, &generated, 42)?;

            if token_id == EOS_TOKEN_ID { break; }
            generated.push(token_id);
            eprintln!("  [token {step}/{len} in {:.1}s]", step_start.elapsed().as_secs_f64(), len = self.max_new_tokens);
        }

        Ok(generated)
    }
}
