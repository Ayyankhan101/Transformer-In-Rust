use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub ffn_dim: usize,
    pub max_seq_len: usize,
    pub dropout: f64,
    pub eps: f64,
    pub blank_ratio: f64,
    pub mask_ratio: f64,
    pub random_replace_ratio: f64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 51200,
            hidden_dim: 256,
            num_layers: 6,
            num_heads: 8,
            ffn_dim: 1024,
            max_seq_len: 512,
            dropout: 0.1,
            eps: 1e-5,
            blank_ratio: 0.15,
            mask_ratio: 0.7,
            random_replace_ratio: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub data_dir: PathBuf,
    pub download_if_empty: bool,
    pub train_split: f64,
    pub seed: u64,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub max_grad_norm: f64,
    pub lr_schedule: LrScheduleConfig,
    pub micro_batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub max_seq_len: usize,
    pub max_steps: usize,
    pub eval_every: usize,
    pub save_every: usize,
    pub log_every: usize,
    pub eval_steps: usize,
    pub checkpoint_dir: PathBuf,
    pub save_optimizer_state: bool,
    pub keep_last_n_checkpoints: usize,
    pub dtype: String,
    pub tokenizer_path: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            download_if_empty: true,
            train_split: 0.95,
            seed: 42,
            learning_rate: 1e-4,
            weight_decay: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            max_grad_norm: 1.0,
            lr_schedule: LrScheduleConfig::default(),
            micro_batch_size: 1,
            gradient_accumulation_steps: 16,
            max_seq_len: 512,
            max_steps: 10000,
            eval_every: 500,
            save_every: 1000,
            log_every: 10,
            eval_steps: 100,
            checkpoint_dir: PathBuf::from("glm_checkpoint"),
            save_optimizer_state: true,
            keep_last_n_checkpoints: 3,
            dtype: "f32".to_string(),
            tokenizer_path: PathBuf::from("codegen_weights/tokenizer.json"),
        }
    }
}

impl TrainingConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LrScheduleConfig {
    #[serde(rename = "type")]
    pub schedule_type: String,
    pub warmup_steps: usize,
    pub max_steps: usize,
    pub min_lr_ratio: f64,
}

impl Default for LrScheduleConfig {
    fn default() -> Self {
        Self {
            schedule_type: "cosine".to_string(),
            warmup_steps: 500,
            max_steps: 10000,
            min_lr_ratio: 0.1,
        }
    }
}

impl TrainConfig {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: TrainConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn to_glm_config(&self) -> crate::glm::config::GLMConfig {
        crate::glm::config::GLMConfig {
            vocab_size: self.model.vocab_size,
            hidden_dim: self.model.hidden_dim,
            num_layers: self.model.num_layers,
            num_heads: self.model.num_heads,
            ffn_dim: self.model.ffn_dim,
            max_seq_len: self.model.max_seq_len,
            _dropout: self.model.dropout,
            eps: self.model.eps,
            blank_ratio: self.model.blank_ratio,
            mask_ratio: self.model.mask_ratio,
            _random_replace_ratio: self.model.random_replace_ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_train_config() -> TrainConfig {
        TrainConfig {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
        }
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert_eq!(config.vocab_size, 51200);
        assert_eq!(config.hidden_dim, 256);
        assert_eq!(config.num_layers, 6);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.ffn_dim, 1024);
        assert_eq!(config.max_seq_len, 512);
        assert_eq!(config.dropout, 0.1);
        assert_eq!(config.eps, 1e-5);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.data_dir, PathBuf::from("data"));
        assert_eq!(config.train_split, 0.95);
        assert_eq!(config.seed, 42);
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.max_grad_norm, 1.0);
        assert_eq!(config.micro_batch_size, 1);
        assert_eq!(config.gradient_accumulation_steps, 16);
        assert_eq!(config.max_steps, 10000);
    }

    #[test]
    fn test_lr_schedule_config_default() {
        let config = LrScheduleConfig::default();
        assert_eq!(config.schedule_type, "cosine");
        assert_eq!(config.warmup_steps, 500);
        assert_eq!(config.max_steps, 10000);
        assert_eq!(config.min_lr_ratio, 0.1);
    }

    #[test]
    fn test_train_config_to_glm_config() {
        let train_config = TrainConfig {
            model: ModelConfig {
                vocab_size: 10000,
                hidden_dim: 128,
                num_layers: 4,
                num_heads: 4,
                ffn_dim: 256,
                max_seq_len: 64,
                dropout: 0.2,
                eps: 1e-6,
                blank_ratio: 0.2,
                mask_ratio: 0.8,
                random_replace_ratio: 0.1,
            },
            training: TrainingConfig::default(),
        };

        let glm_config = train_config.to_glm_config();
        assert_eq!(glm_config.vocab_size, 10000);
        assert_eq!(glm_config.hidden_dim, 128);
        assert_eq!(glm_config.num_layers, 4);
        assert_eq!(glm_config.num_heads, 4);
        assert_eq!(glm_config.ffn_dim, 256);
        assert_eq!(glm_config.max_seq_len, 64);
        assert_eq!(glm_config._dropout, 0.2);
        assert_eq!(glm_config.eps, 1e-6);
    }

    #[test]
    fn test_train_config_serialization() {
        let config = default_train_config();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let deserialized: TrainConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(config.model.vocab_size, deserialized.model.vocab_size);
        assert_eq!(
            config.training.learning_rate,
            deserialized.training.learning_rate
        );
    }

    #[test]
    fn test_train_config_clone() {
        let config = default_train_config();
        let cloned = config.clone();
        assert_eq!(config.model.hidden_dim, cloned.model.hidden_dim);
        assert_eq!(config.training.max_steps, cloned.training.max_steps);
    }

    /// The config the repo ships must actually deserialize. It did not: `eval_steps`
    /// lived under a separate `evaluation:` section and `tokenizer_path` was absent,
    /// and nothing loaded the file because `glm-train` had no `--config` flag.
    #[test]
    fn shipped_config_loads() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/configs/train.yaml");
        let config = TrainConfig::from_file(path).expect("configs/train.yaml must parse");
        assert_eq!(config.model.hidden_dim, 256);
        assert_eq!(config.training.gradient_accumulation_steps, 32);
        assert_eq!(config.training.max_grad_norm, 1.0);
    }

    /// A config naming only a couple of fields should fall back to defaults.
    #[test]
    fn partial_config_falls_back_to_defaults() {
        let yaml = "training:\n  learning_rate: 0.5\n";
        let config: TrainConfig = serde_yaml::from_str(yaml).expect("partial config must parse");
        assert_eq!(config.training.learning_rate, 0.5);
        assert_eq!(
            config.training.micro_batch_size,
            TrainingConfig::default().micro_batch_size
        );
    }
}
