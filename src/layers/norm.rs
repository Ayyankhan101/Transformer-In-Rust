use candle_core::{Device, Result, Tensor};

// --- RMSNorm ---
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::ones(dim, candle_core::DType::F32, device)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let variance = x.sqr()?.mean(last_dim)?;
        let norm_x = (variance + self.eps)?.sqrt()?;
        let norm_x_3d = norm_x.unsqueeze(last_dim)?;
        let weight = if self.weight.dtype() != x.dtype() {
            self.weight.to_dtype(x.dtype())?
        } else {
            self.weight.clone()
        };
        x.broadcast_div(&norm_x_3d)?.broadcast_mul(&weight)
    }

}

// --- LayerNorm ---
pub struct LayerNorm {
    pub weight: Tensor,
    pub bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    #[allow(dead_code)]
    pub fn zeros(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        let weight = Tensor::zeros(dim, candle_core::DType::F32, device)?;
        let bias = Tensor::zeros(dim, candle_core::DType::F32, device)?;
        Ok(Self { weight, bias, eps })
    }

    pub fn zeros_with_dtype(dim: usize, eps: f64, dtype: candle_core::DType, device: &Device) -> Result<Self> {
        let weight = Tensor::zeros(dim, dtype, device)?;
        let bias = Tensor::zeros(dim, dtype, device)?;
        Ok(Self { weight, bias, eps })
    }

    pub fn from_tensor(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let mean = x.mean(last_dim)?;
        let mean = mean.unsqueeze(last_dim)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let variance = x_centered.sqr()?.mean(last_dim)?;
        let std = (variance + self.eps)?.sqrt()?;
        let std = std.unsqueeze(last_dim)?;
        let normalized = x_centered.broadcast_div(&std)?;
        let weight = if self.weight.dtype() != normalized.dtype() {
            self.weight.to_dtype(normalized.dtype())?
        } else {
            self.weight.clone()
        };
        let bias = if self.bias.dtype() != normalized.dtype() {
            self.bias.to_dtype(normalized.dtype())?
        } else {
            self.bias.clone()
        };
        normalized.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_rmsnorm_forward() -> Result<()> {
        let device = Device::Cpu;
        let norm = RMSNorm::new(4, 1e-5, &device)?;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let out = norm.forward(&x)?;
        assert_eq!(out.dims(), &[1, 4]);
        let ratio = out.get(0)?.get(0)?.to_scalar::<f32>()? / 1.0;
        assert!((out.get(0)?.get(1)?.to_scalar::<f32>()? / 2.0 - ratio).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_layernorm_forward() -> Result<()> {
        let device = Device::Cpu;
        let norm = LayerNorm::zeros_with_dtype(4, 1e-5, DType::F32, &device)?;
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &device)?;
        let out = norm.forward(&x)?;
        assert_eq!(out.dims(), &[1, 4]);
        Ok(())
    }
}
