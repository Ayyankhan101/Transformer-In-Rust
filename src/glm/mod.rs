//! GLM (General Language Model) implementation.
//!
//! GLM is a blank-infilling transformer that learns to predict masked spans
//! of text using bidirectional context. It supports:
//!
//! - 2D positional encoding (sequence position + blank position)
//! - Blank-infilling attention masks
//! - Training from scratch on custom data
//! - Safetensors checkpoint save/load

pub mod attention_mask;
pub mod config;
pub mod model;
pub mod positions;
pub mod trainable;

pub use config::GLMConfig;
