#[derive(Debug, Clone)]
pub struct GLMConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub _dropout: f64,
    pub eps: f64,
    pub blank_ratio: f64,
    pub mask_ratio: f64,
    pub _random_replace_ratio: f64,
}

impl Default for GLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 51200,
            hidden_dim: 256,
            num_layers: 6,
            num_heads: 8,
            ffn_dim: 1024,
            max_seq_len: 512,
            _dropout: 0.1,
            eps: 1e-5,
            blank_ratio: 0.15,
            mask_ratio: 0.7,
            _random_replace_ratio: 0.15,
        }
    }
}

impl GLMConfig {
    pub fn param_count_estimate(&self) -> usize {
        let embed = self.vocab_size * self.hidden_dim;
        let pos_embed = self.max_seq_len * self.hidden_dim;
        let per_layer = self.hidden_dim * self.hidden_dim * 4
            + self.hidden_dim * self.ffn_dim * 3
            + self.hidden_dim * 2;
        let lm_head = self.vocab_size * self.hidden_dim;
        embed + pos_embed + per_layer * self.num_layers + lm_head
    }
}
