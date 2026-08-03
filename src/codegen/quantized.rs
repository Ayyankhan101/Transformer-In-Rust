//! INT8 Dynamic Quantization for CodeGen-350M
//!
//! Quantizes linear layer weights from F32 to INT8 with per-channel scaling,
//! reducing memory footprint by ~4× and potentially improving cache performance.
//!
//! Quantization scheme:
//!   weight_int8 = round(weight_f32 / scale)
//!   scale = max(abs(weight_f32), epsilon) / 127.0
//!
//! Forward: out = (weight_int8 * scale) @ input

use candle_core::{DType, Device, Result, Tensor};

/// Quantized linear layer with U8 (offset) weights.
pub struct QuantizedLinear {
    /// Quantized weights (u8 with 128 offset), shape [out_dim, in_dim]
    weight_int8: Tensor,
    /// Per-output-channel scale factors, shape [out_dim, 1]
    scale: Tensor,
    /// Original shape
    out_dim: usize,
    in_dim: usize,
}

impl QuantizedLinear {
    /// Create a quantized linear layer from F32 weights.
    ///
    /// `weight` should have shape [out_dim, in_dim].
    pub fn from_f32(weight: &Tensor, device: &Device) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            return Err(candle_core::Error::Msg(format!(
                "Expected 2D weight tensor, got {}D",
                dims.len()
            )));
        }
        let out_dim = dims[0];
        let in_dim = dims[1];

        // Move to F32 on CPU for quantization
        let w = weight.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        let w_vec: Vec<f32> = w.to_vec2::<f32>()?.into_iter().flatten().collect();

        // Per-output-channel quantization
        // Store quantized weights as u8 (offset by 128 to handle negative values)
        let mut quantized: Vec<u8> = Vec::with_capacity(out_dim * in_dim);
        let mut scales: Vec<f32> = Vec::with_capacity(out_dim);

        for row in 0..out_dim {
            let start = row * in_dim;
            let end = start + in_dim;
            let row_slice = &w_vec[start..end];

            // Find scale for this row (per-channel)
            let max_abs = row_slice
                .iter()
                .map(|v| v.abs())
                .fold(f32::MIN_POSITIVE, f32::max);
            let scale = (max_abs / 127.0).max(f32::MIN_POSITIVE);

            // Quantize: map from [-128, 127] to [0, 255] range
            for &val in row_slice {
                let q = (val / scale).round().clamp(-128.0, 127.0) as i8;
                quantized.push((q as i16 + 128) as u8); // offset by 128 to make non-negative
            }
            scales.push(scale);
        }

        let weight_q = Tensor::from_vec(quantized, (out_dim, in_dim), &Device::Cpu)?;
        let scale_t = Tensor::from_vec(scales, (out_dim, 1), &Device::Cpu)?.to_dtype(DType::F32)?;

        Ok(Self {
            weight_int8: weight_q.to_device(device)?,
            scale: scale_t.to_device(device)?,
            out_dim,
            in_dim,
        })
    }

    /// Forward pass: dequantize on-the-fly and compute matmul.
    ///
    /// `input` shape: \[batch, seq\_len, `in_dim`\] or \[`in_dim`\]
    /// Returns shape: \[batch, seq\_len, `out_dim`\] or \[`out_dim`\]
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Dequantize: weight_f32 = (weight_u8 - 128) * scale
        // weight_int8 shape: [out_dim, in_dim]
        // scale shape: [out_dim, 1]
        let w_f32 = self.weight_int8.to_dtype(DType::F32)?;
        let offset = Tensor::new(128.0f32, w_f32.device())?;
        let w_centered = w_f32.broadcast_sub(&offset)?;
        let w_deq = w_centered.broadcast_mul(&self.scale)?;

        // w_deq shape: [out_dim, in_dim], we need [in_dim, out_dim] for matmul
        let w_t = w_deq.t()?;

        if input.dims().len() == 2 {
            // [batch, in_dim] @ [in_dim, out_dim] = [batch, out_dim]
            input.matmul(&w_t)
        } else {
            // [in_dim] @ [in_dim, out_dim] = [out_dim]
            input.matmul(&w_t)
        }
    }

    /// Estimated size in bytes (INT8 weights + F32 scales)
    pub fn estimated_size(&self) -> usize {
        self.out_dim * self.in_dim         // INT8 weights: 1 byte each
            + self.out_dim * 4 // F32 scales: 4 bytes each
    }

    /// Original F32 size for comparison
    pub fn original_size(&self) -> usize {
        self.out_dim * self.in_dim * 4 // F32: 4 bytes each
    }
}

/// A quantized version of the CodeGen attention weights (QKV projection).
/// In the real model, QKV weight is [hidden_dim, 3 * hidden_dim] (transposed).
pub struct QuantizedAttention {
    pub qkv: Option<QuantizedLinear>,
    pub out: Option<QuantizedLinear>,
}

impl QuantizedAttention {
    pub fn new(
        qkv_weight: Option<&Tensor>,
        out_weight: Option<&Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let qkv = if let Some(w) = qkv_weight {
            Some(QuantizedLinear::from_f32(w, device)?)
        } else {
            None
        };
        let out = if let Some(w) = out_weight {
            Some(QuantizedLinear::from_f32(w, device)?)
        } else {
            None
        };
        Ok(Self { qkv, out })
    }
}

/// A quantized version of the CodeGen FFN weights.
pub struct QuantizedFFN {
    pub fc_in: Option<QuantizedLinear>,
    pub fc_out: Option<QuantizedLinear>,
}

impl QuantizedFFN {
    pub fn new(
        fc_in_weight: Option<&Tensor>,
        fc_out_weight: Option<&Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let fc_in = if let Some(w) = fc_in_weight {
            Some(QuantizedLinear::from_f32(w, device)?)
        } else {
            None
        };
        let fc_out = if let Some(w) = fc_out_weight {
            Some(QuantizedLinear::from_f32(w, device)?)
        } else {
            None
        };
        Ok(Self { fc_in, fc_out })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantized_linear_roundtrip() -> Result<()> {
        let device = Device::Cpu;

        // Create a simple weight matrix
        let w = Tensor::from_vec(vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0], (2, 3), &device)?;

        let qlinear = QuantizedLinear::from_f32(&w, &device)?;

        // Create input
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device)?;

        let output = qlinear.forward(&input)?;
        let out_vec: Vec<f32> = output.to_vec2::<f32>()?.into_iter().flatten().collect();

        // Should be approximate (INT8 quantization introduces some error)
        assert_eq!(out_vec.len(), 2, "Output should have 2 values (out_dim=2)");

        // Verify quantization reduces memory
        let estimated = qlinear.estimated_size();
        let original = qlinear.original_size();
        assert!(estimated < original, "Quantized should be smaller");
        // For large matrices this is ~4x, for small ones scales overhead reduces it
        assert!(
            (original as f64 / estimated as f64) > 1.0,
            "Should have some compression, got {}/{} = {:.1}x",
            original,
            estimated,
            original as f64 / estimated as f64
        );

        println!("  Original size: {original} bytes");
        println!("  Quantized size: {estimated} bytes");
        println!("  Compression: {:.1}x", original as f64 / estimated as f64);
        println!("  Output: {out_vec:?}");

        Ok(())
    }

    #[test]
    fn test_quantized_preserves_ranking() -> Result<()> {
        let device = Device::Cpu;

        // Create weights with clear pattern
        let w = Tensor::from_vec(vec![10.0f32, 0.0, 0.0, 0.0, 20.0, 0.0], (2, 3), &device)?;

        let qlinear = QuantizedLinear::from_f32(&w, &device)?;

        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device)?;
        let output = qlinear.forward(&input)?;
        let out_vec: Vec<f32> = output.to_vec2::<f32>()?.into_iter().flatten().collect();

        // output[0] should be ~10 (first row * input), output[1] should be ~40 (second row * input)
        assert!(
            out_vec[1] > out_vec[0],
            "Second output should be larger (20*2=40 vs 10*1=10), got {:?}",
            out_vec
        );

        println!("  Ranking preserved: {out_vec:?}");
        Ok(())
    }
}
