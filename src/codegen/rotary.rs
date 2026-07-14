use candle_core::{Device, DType, Result, Tensor};

/// Rotary Position Embedding (RoPE)
///
/// Applies rotation in 2D subspaces based on position index.
/// Standard implementation: rotate pairs of dimensions.
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, dtype: DType, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / 10000.0f32.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_tensor = Tensor::new(inv_freq.as_slice(), device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions_tensor = Tensor::new(positions.as_slice(), device)?;

        let freqs = positions_tensor
            .unsqueeze(1)?
            .broadcast_mul(&inv_freq_tensor.unsqueeze(0)?)?;

        let cos_cache = freqs.cos()?.contiguous()?;
        let sin_cache = freqs.sin()?.contiguous()?;

        let cos_cache = cos_cache.to_dtype(dtype)?;
        let sin_cache = sin_cache.to_dtype(dtype)?;

        Ok(Self { cos_cache, sin_cache, dim })
    }

    pub fn apply_rotary(
        &self,
        x: &Tensor,
        positions: &[usize],
    ) -> Result<Tensor> {
        let (_, _, _, head_dim) = x.dims4()?;

        let cos = self.cos_cache.index_select(
            &Tensor::from_slice(
                positions.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
                positions.len(),
                x.device(),
            )?,
            0,
        )?;
        let sin = self.sin_cache.index_select(
            &Tensor::from_slice(
                positions.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(),
                positions.len(),
                x.device(),
            )?,
            0,
        )?;

        let cos = cos.unsqueeze(0)?.unsqueeze(1)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(1)?;

        let rotary_dim = self.dim;

        // Apply rotary to first rotary_dim dimensions using even-odd pairing,
        // matching GPT-J / CodeGen: pair (0,1), (2,3), ..., (rotary_dim-2, rotary_dim-1)
        let x_part = x.narrow(3, 0, rotary_dim)?;
        let x_shape = x_part.dims4()?;
        let x_pairs = x_part.reshape((x_shape.0, x_shape.1, x_shape.2, x_shape.3 / 2, 2))?;
        let x_even = x_pairs.get_on_dim(4, 0)?;
        let x_odd = x_pairs.get_on_dim(4, 1)?;

        let rotated_even = (&x_even.broadcast_mul(&cos)? - &x_odd.broadcast_mul(&sin)?)?;
        let rotated_odd = (&x_odd.broadcast_mul(&cos)? + &x_even.broadcast_mul(&sin)?)?;

        let rotated = Tensor::stack(&[rotated_even, rotated_odd], 4)?;
        let rotated = rotated.reshape(x_shape)?;

        if rotary_dim < head_dim {
            let rest = x.narrow(3, rotary_dim, head_dim - rotary_dim)?;
            Tensor::cat(&[rotated, rest], 3)
        } else {
            Ok(rotated)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_no_segfault() -> Result<()> {
        let device = Device::Cpu;
        let rot = RotaryEmbedding::new(64, 128, DType::F32, &device)?;
        let x = Tensor::randn(0.0f32, 1.0, (1, 4, 8, 64), &device)?;
        let out = rot.apply_rotary(&x, &[0, 1, 2, 3, 4, 5, 6, 7])?;
        assert_eq!(out.dims(), x.dims());
        Ok(())
    }
}
