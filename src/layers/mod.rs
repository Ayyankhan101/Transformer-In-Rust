//! Shared transformer layer implementations.
//!
//! Hand-written building blocks using raw tensor operations from `candle_core`.
//! These layers are shared between GLM and CodeGen models.

pub mod attention;
pub mod block;
pub mod embedding;
pub mod ffn;
pub mod norm;
