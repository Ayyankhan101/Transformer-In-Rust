use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;

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
