use candle_core::{Device, Result, Tensor};

/// GLM 2D Positional Encoding
///
/// Two orthogonal position signals:
/// 1. pos_1: Position of the blank (which blank span)
/// 2. pos_2: Position within the blank (offset inside the span)
///
/// Context tokens get pos_1 = 0, pos_2 = position_in_context
/// Blank tokens get pos_1 = blank_index, pos_2 = offset_in_blank
pub struct GLMPositionEncoding {
    pub pos_1_embedding: Tensor,
    pub pos_2_embedding: Tensor,
    _hidden_dim: usize,
}

impl GLMPositionEncoding {
    pub fn new(max_positions: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        let pos_1_embedding = Tensor::randn(0.0f32, 0.02f32, (max_positions, hidden_dim), device)?;
        let pos_2_embedding = Tensor::randn(0.0f32, 0.02f32, (max_positions, hidden_dim), device)?;
        Ok(Self {
            pos_1_embedding,
            pos_2_embedding,
            _hidden_dim: hidden_dim,
        })
    }

    /// Build 2D position encoding for a sequence with blanks
    ///
    /// Args:
    /// - context_len: Length of the context segment
    /// - blank_lens: Lengths of each blank segment
    ///
    /// Returns tensor of shape [1, total_seq_len, hidden_dim]
    pub fn forward(
        &self,
        context_len: usize,
        blank_lens: &[usize],
        device: &Device,
    ) -> Result<Tensor> {
        let total_len = context_len + blank_lens.iter().sum::<usize>();

        let mut pos_1_ids = Vec::new();
        let mut pos_2_ids = Vec::new();

        // Context tokens: pos_1 = 0 (context), pos_2 = position in context
        for i in 0..context_len {
            pos_1_ids.push(0u32);
            pos_2_ids.push(i as u32);
        }

        // Blank tokens
        for (blank_idx, &blank_len) in blank_lens.iter().enumerate() {
            for local_offset in 0..blank_len {
                pos_1_ids.push((blank_idx + 1) as u32);
                pos_2_ids.push(local_offset as u32);
            }
        }

        let pos_1_tensor = Tensor::from_slice(&pos_1_ids, total_len, device)?;
        let pos_2_tensor = Tensor::from_slice(&pos_2_ids, total_len, device)?;

        let pos_1_emb = self.pos_1_embedding.index_select(&pos_1_tensor, 0)?;
        let pos_2_emb = self.pos_2_embedding.index_select(&pos_2_tensor, 0)?;

        // Combine: add both 2D positional signals
        let combined = pos_1_emb.broadcast_add(&pos_2_emb)?;
        combined.unsqueeze(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_encoding_new() {
        let device = Device::Cpu;
        let pos_enc = GLMPositionEncoding::new(128, 256, &device).unwrap();
        assert_eq!(pos_enc.pos_1_embedding.dims(), &[128, 256]);
        assert_eq!(pos_enc.pos_2_embedding.dims(), &[128, 256]);
    }

    #[test]
    fn test_position_encoding_context_only() {
        let device = Device::Cpu;
        let pos_enc = GLMPositionEncoding::new(64, 128, &device).unwrap();

        // Context only (no blanks)
        let result = pos_enc.forward(4, &[], &device).unwrap();
        assert_eq!(result.dims(), &[1, 4, 128]);
    }

    #[test]
    fn test_position_encoding_with_blanks() {
        let device = Device::Cpu;
        let pos_enc = GLMPositionEncoding::new(64, 128, &device).unwrap();

        // Context of length 3, blanks of length [2, 1]
        let result = pos_enc.forward(3, &[2, 1], &device).unwrap();
        // Total: 3 + 2 + 1 = 6
        assert_eq!(result.dims(), &[1, 6, 128]);
    }

    #[test]
    fn test_position_encoding_shape_consistency() {
        let device = Device::Cpu;
        let pos_enc = GLMPositionEncoding::new(32, 64, &device).unwrap();

        // Different blank configurations
        let r1 = pos_enc.forward(5, &[3], &device).unwrap();
        let r2 = pos_enc.forward(2, &[1, 2], &device).unwrap();

        assert_eq!(r1.dims(), &[1, 8, 64]); // 5 + 3
        assert_eq!(r2.dims(), &[1, 5, 64]); // 2 + 1 + 2
    }

    #[test]
    fn test_position_encoding_different_dims() {
        let device = Device::Cpu;
        let pos_enc = GLMPositionEncoding::new(16, 32, &device).unwrap();
        let result = pos_enc.forward(2, &[2], &device).unwrap();
        assert_eq!(result.dims(), &[1, 4, 32]);
    }
}
