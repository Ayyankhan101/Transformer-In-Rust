use anyhow::Result;
use tokenizers::Tokenizer as HFTokenizer;

pub struct CodeGenTokenizer {
    tokenizer: HFTokenizer,
}

impl CodeGenTokenizer {
    pub fn from_file(path: &str) -> Result<Self> {
        let tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoded = self.tokenizer.encode(text, false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(encoded.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }
}