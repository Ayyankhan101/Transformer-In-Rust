//! Code generation pipeline for CodeGen-350M and GLM models.
//!
//! Provides autoregressive generation with:
//! - Token streaming ([`codegen_generate::CodeGenGenerator`])
//! - Multi-turn chat sessions ([`chat::ChatSession`])
//! - Sampling with repetition penalty, temperature, top-k, and top-p

pub mod chat;
pub mod codegen_generate;
pub mod glm_generate;
pub mod sampling;
