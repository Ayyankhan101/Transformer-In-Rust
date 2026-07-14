use candle_core::{Device, Result, Tensor};

#[allow(dead_code)]
pub enum Activation {
    SwiGLU,
    GELU,
    GELUNew,
}

fn gelu_new(x: &Tensor) -> Result<Tensor> {
    let sqrt_2_over_pi = (2.0f64 / std::f64::consts::PI).sqrt();
    let x_cubed = (x * x * x)?;
    let inner = ((&x_cubed * 0.044715)? + x)?;
    let inner = (inner * sqrt_2_over_pi)?;
    let tanh_inner = tanh_approx(&inner)?;
    let one_plus_tanh = (tanh_inner + 1.0)?;
    (x * one_plus_tanh)? * 0.5
}

fn tanh_approx(x: &Tensor) -> Result<Tensor> {
    let twice = (x * 2.0)?;
    let sig = candle_nn::ops::sigmoid(&twice)?;
    (sig * 2.0)? - 1.0
}

/// SwiGLU gate: gate(x) = x * sigmoid(x)  (Swish activation)
pub fn swiglu_gate(gate: &Tensor) -> Result<Tensor> {
    gate * candle_nn::ops::sigmoid(gate)
}

// --- SwiGLU FFN ---
pub struct SwiGLU {
    gate: Tensor,
    up: Tensor,
    down: Tensor,
}

impl SwiGLU {
    pub fn new_blank(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let gate = Tensor::zeros((hidden_dim, ffn_dim), candle_core::DType::F32, device)?;
        let up = Tensor::zeros((hidden_dim, ffn_dim), candle_core::DType::F32, device)?;
        let down = Tensor::zeros((ffn_dim, hidden_dim), candle_core::DType::F32, device)?;
        Ok(Self { gate, up, down })
    }

    pub fn new(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let gate = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), device)?;
        let up = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), device)?;
        let down = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), device)?;
        Ok(Self { gate, up, down })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_out = x.broadcast_matmul(&self.gate.unsqueeze(0)?)?;
        let swish = (&gate_out * candle_nn::ops::sigmoid(&gate_out)?)?;
        let up_out = x.broadcast_matmul(&self.up.unsqueeze(0)?)?;
        let hidden = (&swish * up_out)?;
        hidden.broadcast_matmul(&self.down.unsqueeze(0)?)
    }
}

// --- GELU FFN (CodeGen-style) ---
pub struct GELUFFN {
    fc_in: Tensor,
    fc_out: Tensor,
}

impl GELUFFN {
    pub fn new_blank(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let fc_in = Tensor::zeros((hidden_dim, ffn_dim), candle_core::DType::F32, device)?;
        let fc_out = Tensor::zeros((ffn_dim, hidden_dim), candle_core::DType::F32, device)?;
        Ok(Self { fc_in, fc_out })
    }

    pub fn new(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let fc_in = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), device)?;
        let fc_out = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), device)?;
        Ok(Self { fc_in, fc_out })
    }

    #[allow(dead_code)]
    pub fn from_tensors(fc_in: Tensor, fc_out: Tensor) -> Self {
        Self { fc_in, fc_out }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.broadcast_matmul(&self.fc_in.unsqueeze(0)?)?;
        let activated = hidden.gelu()?;
        activated.broadcast_matmul(&self.fc_out.unsqueeze(0)?)
    }
}

// --- GELU New FFN (CodeGen "gelu_new" variant, tanh-based) ---
pub struct GELUNewFFN {
    fc_in: Tensor,
    fc_in_bias: Tensor,
    fc_out: Tensor,
    fc_out_bias: Tensor,
}

impl GELUNewFFN {
    pub fn new_blank(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let fc_in = Tensor::zeros((hidden_dim, ffn_dim), candle_core::DType::F32, device)?;
        let fc_in_bias = Tensor::zeros(ffn_dim, candle_core::DType::F32, device)?;
        let fc_out = Tensor::zeros((ffn_dim, hidden_dim), candle_core::DType::F32, device)?;
        let fc_out_bias = Tensor::zeros(hidden_dim, candle_core::DType::F32, device)?;
        Ok(Self { fc_in, fc_in_bias, fc_out, fc_out_bias })
    }

    #[allow(dead_code)]
    pub fn from_tensors(fc_in: Tensor, fc_out: Tensor) -> Self {
        let ffn_dim = fc_in.dim(1).unwrap_or(0);
        let hidden_dim = fc_out.dim(1).unwrap_or(0);
        let device = fc_in.device();
        let dtype = fc_in.dtype();
        let fc_in_bias = Tensor::zeros(ffn_dim, dtype, device).unwrap();
        let fc_out_bias = Tensor::zeros(hidden_dim, dtype, device).unwrap();
        Self { fc_in, fc_in_bias, fc_out, fc_out_bias }
    }

    pub fn from_tensors_with_bias(fc_in: Tensor, fc_in_bias: Tensor, fc_out: Tensor, fc_out_bias: Tensor) -> Self {
        Self { fc_in, fc_in_bias, fc_out, fc_out_bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.broadcast_matmul(&self.fc_in.unsqueeze(0)?)?;
        let hidden = hidden.broadcast_add(&self.fc_in_bias)?;
        let activated = gelu_new(&hidden)?;
        let out = activated.broadcast_matmul(&self.fc_out.unsqueeze(0)?)?;
        out.broadcast_add(&self.fc_out_bias)
    }
}

// --- Configurable FFN ---
pub enum FeedForward {
    SwiGLU(SwiGLU),
    GELU(GELUFFN),
    GELUNew(GELUNewFFN),
}

impl FeedForward {
    pub fn new_blank(activation: Activation, hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        match activation {
            Activation::SwiGLU => Ok(Self::SwiGLU(SwiGLU::new_blank(hidden_dim, ffn_dim, device)?)),
            Activation::GELU => Ok(Self::GELU(GELUFFN::new_blank(hidden_dim, ffn_dim, device)?)),
            Activation::GELUNew => Ok(Self::GELUNew(GELUNewFFN::new_blank(hidden_dim, ffn_dim, device)?)),
        }
    }

    pub fn new(activation: Activation, hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        match activation {
            Activation::SwiGLU => Ok(Self::SwiGLU(SwiGLU::new(hidden_dim, ffn_dim, device)?)),
            Activation::GELU => Ok(Self::GELU(GELUFFN::new(hidden_dim, ffn_dim, device)?)),
            Activation::GELUNew => Ok(Self::GELUNew(GELUNewFFN::new_blank(hidden_dim, ffn_dim, device)?)),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::SwiGLU(f) => f.forward(x),
            Self::GELU(f) => f.forward(x),
            Self::GELUNew(f) => f.forward(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swiglu_gate() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::new(&[[0.0f32, 1.0, -1.0, 2.0]], &device)?;
        let out = swiglu_gate(&x)?;
        // For x=1.0: 1.0 * sigmoid(1.0) ≈ 1.0 * 0.731 = 0.731
        let v = out.to_vec2::<f32>()?;
        assert!((v[0][1] - 0.731).abs() < 0.01);
        // For x=0.0: 0.0 * 0.5 = 0.0
        assert!((v[0][0] - 0.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_gelu_forward() -> Result<()> {
        let device = Device::Cpu;
        let ffn = FeedForward::new(Activation::GELU, 8, 32, &device)?;
        let x = Tensor::randn(0.0f32, 1.0f32, (1, 3, 8), &device)?;
        let out = ffn.forward(&x)?;
        assert_eq!(out.dims(), &[1, 3, 8]);
        Ok(())
    }
}
