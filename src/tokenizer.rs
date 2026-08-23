use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;

#[derive(Clone)]
pub struct CodeGenTokenizer {
    tokenizer: HFTokenizer,
}

impl CodeGenTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoded.get_ids().to_vec())
    }

    /// Number of ids the tokenizer can actually decode, including added tokens.
    ///
    /// CodeGen's `config.json` declares `vocab_size: 51200`, but the tokenizer tops
    /// out at id 50294 — the embedding and `lm_head` are padded to a round number.
    /// Those padding rows are untrained, so their logits are arbitrary and sampling
    /// can land on an id that decodes to nothing.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_tokenizer_from_invalid_path() {
        let result = CodeGenTokenizer::from_file("nonexistent_tokenizer.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_empty_string() {
        if !Path::new("codegen_weights/tokenizer.json").exists() {
            eprintln!("Skipping: tokenizer not found");
            return;
        }
        let tok = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json").unwrap();
        let ids = tok.encode("").unwrap();
        assert!(ids.is_empty() || ids.len() <= 1);
    }

    #[test]
    fn test_decode_empty_ids() {
        if !Path::new("codegen_weights/tokenizer.json").exists() {
            eprintln!("Skipping: tokenizer not found");
            return;
        }
        let tok = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json").unwrap();
        let text = tok.decode(&[]).unwrap();
        assert!(text.is_empty());
    }

    /// The model's declared `vocab_size` is padded well beyond what the tokenizer
    /// can decode, which is why generation trims logits before sampling.
    #[test]
    fn test_vocab_size_is_below_the_models_padded_vocab() {
        if !Path::new("codegen_weights/tokenizer.json").exists() {
            eprintln!("Skipping: tokenizer not found");
            return;
        }
        let tok = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json").unwrap();
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string("codegen_weights/config.json").unwrap())
                .unwrap();
        let declared = config["vocab_size"].as_u64().unwrap() as usize;

        assert!(
            tok.vocab_size() < declared,
            "expected padding: tokenizer {} vs config {declared}",
            tok.vocab_size()
        );
    }

    #[test]
    fn test_encode_special_characters() {
        if !Path::new("codegen_weights/tokenizer.json").exists() {
            eprintln!("Skipping: tokenizer not found");
            return;
        }
        let tok = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json").unwrap();
        let ids = tok.encode("<|endoftext|>").unwrap();
        assert!(!ids.is_empty());
    }
}
