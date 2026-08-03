use candle_core::Result;

use crate::codegen::kv_cache::KVCache;
use crate::codegen::model::CodeGenModel;
use crate::generation::sampling::sample;

const EOS_TOKEN_ID: u32 = 50256;

// ── S6.1: Token Streaming ─────────────────────────────────────────────────

/// Callback trait for token streaming.
///
/// Return `false` to stop generation early.
pub trait StreamHandler {
    fn on_token(&mut self, token: u32, text: &str) -> bool;
}

/// A simple handler that prints tokens to stdout as they're generated.
pub struct PrintStream;

impl StreamHandler for PrintStream {
    fn on_token(&mut self, _token: u32, text: &str) -> bool {
        print!("{text}");
        use std::io::Write;
        let _ = std::io::stdout().flush();
        true // continue generation
    }
}

/// A handler that collects all tokens into a Vec (non-streaming fallback).
pub struct CollectStream {
    pub tokens: Vec<u32>,
}

impl Default for CollectStream {
    fn default() -> Self {
        Self::new()
    }
}

impl CollectStream {
    pub fn new() -> Self {
        Self { tokens: Vec::new() }
    }
}

impl StreamHandler for CollectStream {
    fn on_token(&mut self, token: u32, _text: &str) -> bool {
        self.tokens.push(token);
        true
    }
}

// ── S6.2: Prompt Templates ─────────────────────────────────────────────────

/// Prompt template variants for CodeGen generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptTemplate {
    /// Plain completion: just the raw prompt text
    Completion,
    /// Instruct format: "instruction\n\nprompt\n\n### Response\n"
    Instruct,
    /// Chat format with system/user/assistant turns
    Chat,
}

impl PromptTemplate {
    /// Apply the template to a raw prompt string, returning the formatted prompt.
    pub fn apply(&self, prompt: &str) -> String {
        match self {
            PromptTemplate::Completion => prompt.to_string(),
            PromptTemplate::Instruct => {
                format!("{prompt}\n\n### Response\n")
            }
            PromptTemplate::Chat => {
                format!("<|user|>\n{prompt}\n<|assistant|>\n")
            }
        }
    }
}

// ── Generator ──────────────────────────────────────────────────────────────

pub struct CodeGenGenerator {
    model: CodeGenModel,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
    max_new_tokens: usize,
    /// Tokenizer for decoding tokens to text for the streaming callback.
    /// If None, empty strings are passed to the callback.
    tokenizer: Option<crate::tokenizer::CodeGenTokenizer>,
    /// Prompt template to apply
    template: PromptTemplate,
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
        Self {
            model,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            max_new_tokens,
            tokenizer: None,
            template: PromptTemplate::Completion,
        }
    }

    /// Attach a tokenizer for text decoding in streaming callbacks.
    pub fn with_tokenizer(mut self, tokenizer: crate::tokenizer::CodeGenTokenizer) -> Self {
        self.tokenizer = Some(tokenizer);
        self
    }

    /// Set the prompt template.
    pub fn with_template(mut self, template: PromptTemplate) -> Self {
        self.template = template;
        self
    }

    pub fn set_temperature(&mut self, t: f64) {
        self.temperature = t;
    }
    pub fn set_top_k(&mut self, k: usize) {
        self.top_k = k;
    }
    pub fn set_top_p(&mut self, p: f64) {
        self.top_p = p;
    }
    pub fn set_max_new_tokens(&mut self, n: usize) {
        self.max_new_tokens = n;
    }
    pub fn set_repetition_penalty(&mut self, p: f64) {
        self.repetition_penalty = p;
    }

    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    pub fn max_new_tokens(&self) -> usize {
        self.max_new_tokens
    }

    // ── Standard generate (collects all tokens) ──

    pub fn generate(&self, prompt_token_ids: &[u32]) -> Result<Vec<u32>> {
        let mut collector = CollectStream::new();
        self.generate_stream(prompt_token_ids, &mut collector)?;
        Ok(collector.tokens)
    }

    // ── S6.1: Stream generate ──

    /// Generate tokens, calling `handler.on_token()` for each new token.
    /// The handler can return `false` to stop generation early.
    pub fn generate_stream(
        &self,
        prompt_token_ids: &[u32],
        handler: &mut dyn StreamHandler,
    ) -> Result<()> {
        let gen_start = std::time::Instant::now();

        let mut cache: Option<Vec<KVCache>> = None;
        let positions: Vec<usize> = (0..prompt_token_ids.len()).collect();
        let logits = self
            .model
            .forward_with_cache(prompt_token_ids, &positions, &mut cache)?;

        let last_logits = logits.get(0)?.get(prompt_token_ids.len() - 1)?;
        let first_token = sample(
            &last_logits,
            self.temperature,
            self.top_k,
            self.top_p,
            self.repetition_penalty,
            prompt_token_ids,
            42,
        )?;

        let mut generated = prompt_token_ids.to_vec();
        generated.push(first_token);

        // Notify handler about the first token
        let first_text = self.decode_token(first_token);
        if !handler.on_token(first_token, &first_text) {
            return Ok(());
        }

        eprintln!(
            "  [prefill done in {:.1}s]",
            gen_start.elapsed().as_secs_f64()
        );

        for step in 1..self.max_new_tokens {
            let step_start = std::time::Instant::now();
            let input_id = [generated.last().copied().unwrap()];
            let pos = vec![generated.len() - 1];

            let logits = self.model.forward_with_cache(&input_id, &pos, &mut cache)?;
            let token_logits = logits.get(0)?.get(0)?;

            let token_id = sample(
                &token_logits,
                self.temperature,
                self.top_k,
                self.top_p,
                self.repetition_penalty,
                &generated,
                42,
            )?;

            if token_id == EOS_TOKEN_ID {
                break;
            }

            generated.push(token_id);

            // Notify handler
            let text = self.decode_token(token_id);
            if !handler.on_token(token_id, &text) {
                break; // handler requested early stop
            }

            eprintln!(
                "  [token {step}/{len} in {:.1}s]",
                step_start.elapsed().as_secs_f64(),
                len = self.max_new_tokens
            );
        }

        Ok(())
    }

    /// Decode a single token ID to string, if a tokenizer is attached.
    fn decode_token(&self, token_id: u32) -> String {
        self.tokenizer
            .as_ref()
            .and_then(|tok| tok.decode(&[token_id]).ok())
            .unwrap_or_default()
    }

    /// Get a reference to the model.
    pub fn model(&self) -> &CodeGenModel {
        &self.model
    }

    // ── S6.4: Batched Generation ──

    /// Generate from multiple prompts in a single forward pass.
    ///
    /// Each prompt is processed independently with its own KV cache.
    /// Returns one `Vec<u32>` per prompt.
    ///
    /// Note: This is a sequential-over-batches implementation (not true
    /// parallel batched attention), but it provides the correct batched
    /// generation API. True parallel batching would require modifying
    /// the attention to handle batch dimensions in the Q/K/V matmuls.
    pub fn generate_batched(&self, prompts: &[&[u32]]) -> Result<Vec<Vec<u32>>> {
        let mut results = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let tokens = self.generate(prompt)?;
            results.push(tokens);
        }

        Ok(results)
    }

    /// Generate from multiple prompts with streaming callbacks.
    ///
    /// Each prompt gets its own handler.
    pub fn generate_batched_stream(
        &self,
        prompts: &[&[u32]],
        handlers: &mut [&mut dyn StreamHandler],
    ) -> Result<()> {
        if prompts.len() != handlers.len() {
            return Err(candle_core::Error::Msg(format!(
                "Number of prompts ({}) must match number of handlers ({})",
                prompts.len(),
                handlers.len()
            )));
        }

        for (i, prompt) in prompts.iter().enumerate() {
            self.generate_stream(prompt, &mut *handlers[i])?;
        }

        Ok(())
    }
}
