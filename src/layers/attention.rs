use candle_core::{Device, Result, Tensor};

pub struct MultiHeadAttention {
    qkv_weight: Tensor,
    out_weight: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new_blank(hidden_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let qkv_weight = Tensor::zeros((hidden_dim, hidden_dim * 3), candle_core::DType::F32, device)?;
        let out_weight = Tensor::zeros((hidden_dim, hidden_dim), candle_core::DType::F32, device)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self { qkv_weight, out_weight, num_heads, head_dim, scale })
    }

    pub fn new(hidden_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        assert_eq!(hidden_dim % num_heads, 0);
        let head_dim = hidden_dim / num_heads;
        let qkv_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), device)?;
        let out_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), device)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Ok(Self { qkv_weight, out_weight, num_heads, head_dim, scale })
    }

    pub fn from_tensors(qkv_weight: Tensor, out_weight: Tensor, num_heads: usize) -> Self {
        let hidden_dim = qkv_weight.dim(0).unwrap();
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        Self { qkv_weight, out_weight, num_heads, head_dim, scale }
    }

    pub fn forward_with_mask(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_dim) = x.dims3()?;
        let qkv = x.broadcast_matmul(&self.qkv_weight.unsqueeze(0)?)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((0, 3, 2, 1, 4))?; // [batch, heads, 3, seq, head_dim]
        let q = qkv.get_on_dim(2, 0)?;
        let k = qkv.get_on_dim(2, 1)?;
        let v = qkv.get_on_dim(2, 2)?;

        let scores = q.broadcast_matmul(&k.transpose(2, 3)?)?;
        let scaled_scores = (scores * self.scale)?;

        let scaled_scores = if let Some(m) = mask {
            if m.dtype() != scaled_scores.dtype() {
                scaled_scores.broadcast_add(&m.to_dtype(scaled_scores.dtype())?)?
            } else {
                scaled_scores.broadcast_add(m)?
            }
        } else {
            scaled_scores
        };

        let attention_weights = candle_nn::ops::softmax(&scaled_scores, 3)?;
        let context = attention_weights.broadcast_matmul(&v)?;

        let context = context.permute((0, 2, 1, 3))?.reshape((batch_size, seq_len, hidden_dim))?;
        context.broadcast_matmul(&self.out_weight.unsqueeze(0)?)
    }

    pub fn qkv_weight(&self) -> &Tensor {
        &self.qkv_weight
    }

    pub fn out_weight(&self) -> &Tensor {
        &self.out_weight
    }
}

// --- Attention Masks ---

pub fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..seq_len)
                .map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 })
        })
        .collect();
    Tensor::from_slice(&mask, (1, 1, seq_len, seq_len), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_shape() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_mask(4, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        Ok(())
    }

    #[test]
    fn test_causal_mask_triangular() -> Result<()> {
        let device = Device::Cpu;
        let mask = causal_mask(3, &device)?;
        // mask shape [1, 1, 3, 3]: extract batch=0, head=0 -> [3, 3]
        let m = mask.get(0)?.get(0)?.to_vec2::<f32>()?;
        assert_eq!(m[0][0], 0.0);
        assert_eq!(m[0][1], f32::NEG_INFINITY);
        assert_eq!(m[0][2], f32::NEG_INFINITY);
        assert_eq!(m[1][0], 0.0);
        assert_eq!(m[1][1], 0.0);
        assert_eq!(m[1][2], f32::NEG_INFINITY);
        assert_eq!(m[2][0], 0.0);
        assert_eq!(m[2][1], 0.0);
        assert_eq!(m[2][2], 0.0);
        Ok(())
    }
}


