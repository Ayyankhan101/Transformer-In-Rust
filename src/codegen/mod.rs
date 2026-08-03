//! CodeGen-350M model implementation.
//!
//! Salesforce's CodeGen-350M is a causal decoder-only transformer for code generation.
//! This module provides the model architecture, weight loading, and inference utilities.
//!
//! # Architecture
//!
//! - 20 transformer layers with hidden_dim=1024, 16 heads
//! - Rotary Position Embedding (RoPE)
//! - KV cache for efficient autoregressive generation
//! - Parallel attention + FFN (GPT-J style)
//! - INT8 quantization support

pub mod config;
pub mod kv_cache;
pub mod model;
pub mod quantized;
pub mod rotary;
pub mod weights;
