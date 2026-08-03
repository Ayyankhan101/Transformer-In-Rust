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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = GLMConfig::default();
        assert_eq!(config.vocab_size, 51200);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.ffn_dim, 1024);
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.eps, 1e-5);
    }

    #[test]
    fn test_param_count_estimate() {
        let config = GLMConfig::default();
        let params = config.param_count_estimate();
        // Should be a reasonable number (millions)
        assert!(params > 1_000_000);
        assert!(params < 100_000_000);
    }

    #[test]
    fn test_param_count_small_model() {
        let config = GLMConfig {
            vocab_size: 1000,
            hidden_dim: 64,
            num_layers: 2,
            num_heads: 4,
            ffn_dim: 128,
            max_seq_len: 32,
            ..Default::default()
        };
        let params = config.param_count_estimate();
        // Small model should have fewer params
        assert!(params < 10_000_000);
    }

    #[test]
    fn test_config_clone() {
        let config = GLMConfig::default();
        let cloned = config.clone();
        assert_eq!(config.vocab_size, cloned.vocab_size);
        assert_eq!(config.hidden_dim, cloned.hidden_dim);
        assert_eq!(config.num_layers, cloned.num_layers);
    }
}
