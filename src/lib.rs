//! Pure Rust transformer implementation for GLM blank-infilling and CodeGen.
//!
//! This crate provides:
//! - GLM (General Language Model) architecture for infilling
//! - CodeGen-350M architecture for code generation
//! - FP16 inference
//! - CLI subcommand implementations
//! - HTTP server for model serving (feature-gated)
//! - Training pipeline with YAML config
//!
//! # Features
//!
//! - `server` — Enables the HTTP server module

pub mod cli;
pub mod codegen;
pub mod commands;
pub mod generation;
pub mod glm;
pub mod layers;
pub mod model;
#[cfg(feature = "server")]
pub mod server;
pub mod tokenizer;
pub mod training;
