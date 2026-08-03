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

#[allow(clippy::upper_case_acronyms)]
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
        Ok(Self {
            norm1,
            attn,
            norm2,
            ffn,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward_with_mask(&normed, mask)?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        x.broadcast_add(&ffn_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::attention::causal_mask;

    #[test]
    fn test_transformer_block_new() {
        let device = Device::Cpu;
        let block =
            TransformerBlock::new(128, 4, 256, NormType::RMSNorm, Activation::GELU, &device)
                .unwrap();
        // Block created successfully
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 4, 128), &device).unwrap();
        let output = block.forward(&x, None).unwrap();
        assert_eq!(output.dims(), &[1, 4, 128]);
    }

    #[test]
    fn test_transformer_block_forward_no_mask() {
        let device = Device::Cpu;
        let block = TransformerBlock::new(64, 4, 128, NormType::RMSNorm, Activation::GELU, &device)
            .unwrap();

        let x = Tensor::randn(0.0f32, 1.0f32, (1, 8, 64), &device).unwrap();
        let output = block.forward(&x, None).unwrap();
        assert_eq!(output.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_transformer_block_forward_with_mask() {
        let device = Device::Cpu;
        let block = TransformerBlock::new(64, 4, 128, NormType::RMSNorm, Activation::GELU, &device)
            .unwrap();

        let x = Tensor::randn(0.0f32, 1.0f32, (1, 8, 64), &device).unwrap();
        let mask = causal_mask(8, &device, candle_core::DType::F32).unwrap();
        let output = block.forward(&x, Some(&mask)).unwrap();
        assert_eq!(output.dims(), &[1, 8, 64]);
    }

    #[test]
    fn test_transformer_block_swiglu() {
        let device = Device::Cpu;
        let block =
            TransformerBlock::new(64, 4, 128, NormType::RMSNorm, Activation::SwiGLU, &device)
                .unwrap();

        let x = Tensor::randn(0.0f32, 1.0f32, (1, 4, 64), &device).unwrap();
        let output = block.forward(&x, None).unwrap();
        assert_eq!(output.dims(), &[1, 4, 64]);
    }

    #[test]
    fn test_transformer_block_different_sizes() {
        let device = Device::Cpu;

        // Small block
        let block1 =
            TransformerBlock::new(32, 2, 64, NormType::RMSNorm, Activation::GELU, &device).unwrap();
        let x1 = Tensor::randn(0.0f32, 1.0f32, (1, 4, 32), &device).unwrap();
        let out1 = block1.forward(&x1, None).unwrap();
        assert_eq!(out1.dims(), &[1, 4, 32]);

        // Larger block
        let block2 =
            TransformerBlock::new(256, 8, 512, NormType::RMSNorm, Activation::GELU, &device)
                .unwrap();
        let x2 = Tensor::randn(0.0f32, 1.0f32, (1, 16, 256), &device).unwrap();
        let out2 = block2.forward(&x2, None).unwrap();
        assert_eq!(out2.dims(), &[1, 16, 256]);
    }
}
