#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub training: TrainingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
        }
    }
}

impl TrainingConfig {
    pub fn to_glm_config(&self) -> crate::glm::config::GLMConfig {
        crate::glm::config::GLMConfig {
            vocab_size: 51200,
            hidden_dim: 256,
            num_layers: 6,
            num_heads: 8,
            ffn_dim: 1024,
            max_seq_len: self.max_seq_len,
            _dropout: 0.1,
            eps: 1e-5,
            blank_ratio: 0.15,
            mask_ratio: 0.7,
            _random_replace_ratio: 0.15,
        }
    }

    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: TrainingConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
