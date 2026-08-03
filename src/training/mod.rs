//! Training pipeline for GLM (General Language Model).
//!
//! Provides a complete training workflow including:
//! - YAML-based configuration ([`config::TrainConfig`])
//! - Data loading with train/eval split ([`data::DataLoader`])
//! - Learning rate scheduling with warmup ([`lr_scheduler::LrScheduler`])
//! - Gradient accumulation and clipping
//! - Safetensors checkpoint save/load

pub mod config;
pub mod data;
pub mod lr_scheduler;
pub mod train;

pub use config::TrainConfig;
pub use config::TrainingConfig;
#[allow(unused_imports)]
pub use data::DataLoader;
#[allow(unused_imports)]
pub use lr_scheduler::LrScheduler;
pub use train::GLMTrainer;
