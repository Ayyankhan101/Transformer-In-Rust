use candle_core::{Device, Result, Tensor};

use super::attention::MultiHeadAttention;
use super::ffn::{Activation, FeedForward};
use super::norm::RMSNorm;

pub enum NormType {
    RMSNorm,
}

pub struct TransformerBlock {
    pub norm1: NormVariant,
    pub attn: MultiHeadAttention,
    pub norm2: NormVariant,
    pub ffn: FeedForward,
}

pub enum NormVariant {
    RMS(RMSNorm),
}

impl NormVariant {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::RMS(n) => n.forward(x),
        }
    }
}

impl TransformerBlock {
    pub fn new(
        hidden_dim: usize,
        num_heads: usize,
        ffn_dim: usize,
        norm_type: NormType,
        activation: Activation,
        device: &Device,
    ) -> Result<Self> {
        let norm1 = match norm_type {
            NormType::RMSNorm => NormVariant::RMS(RMSNorm::new(hidden_dim, 1e-5, device)?),
        };
        let norm2 = match norm_type {
            NormType::RMSNorm => NormVariant::RMS(RMSNorm::new(hidden_dim, 1e-5, device)?),
        };
        let attn = MultiHeadAttention::new(hidden_dim, num_heads, device)?;
        let ffn = FeedForward::new(activation, hidden_dim, ffn_dim, device)?;
        Ok(Self { norm1, attn, norm2, ffn })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward_with_mask(&normed, mask)?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        x.broadcast_add(&ffn_out)
    }
}
