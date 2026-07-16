use candle_core::DType;

#[derive(Debug, Clone)]
pub struct CodeGenConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub rotary_dim: usize,
    pub eps: f64,
    pub _initializer_range: f32,
    pub dtype: DType,
}

impl Default for CodeGenConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50400,
            hidden_dim: 1024,
            num_layers: 20,
            num_heads: 16,
            ffn_dim: 4096,
            max_seq_len: 2048,
            rotary_dim: 64,
            eps: 1e-5,
            _initializer_range: 0.02,
            dtype: DType::F32,
        }
    }
}

impl CodeGenConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    pub fn from_hf_config(config: &serde_json::Value) -> Self {
        let vocab_size = config["vocab_size"].as_u64().unwrap_or(50400) as usize;
        let hidden_dim = config["n_embd"].as_u64().unwrap_or(1024) as usize;
        let num_layers = config["n_layer"].as_u64().unwrap_or(20) as usize;
        let num_heads = config["n_head"].as_u64().unwrap_or(16) as usize;
        let ffn_dim = config["n_inner"]
            .as_u64()
            .map(|v| v as usize)
            .unwrap_or(hidden_dim * 4);
        let max_seq_len = config["n_positions"].as_u64().unwrap_or(2048) as usize;
        let rotary_dim = config["rotary_dim"].as_u64().unwrap_or(64) as usize;

        Self {
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            ffn_dim,
            max_seq_len,
            rotary_dim,
            eps: 1e-5,
            _initializer_range: 0.02,
            dtype: DType::F32,
        }
    }
}
