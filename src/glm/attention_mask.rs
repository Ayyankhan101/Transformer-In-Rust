use candle_core::{Device, Result, Tensor};

/// Build the GLM blank-infilling attention mask.
///
/// Rules:
/// - Context tokens attend to all context + all blanks (bidirectional)
/// - Blank tokens attend to all context + earlier blanks + same blank (causal within span)
/// - Later blanks are masked out for earlier blank tokens
pub fn build_glm_mask(
    context_len: usize,
    blank_lens: &[usize],
    device: &Device,
) -> Result<Tensor> {
    let total_len = context_len + blank_lens.iter().sum::<usize>();
    let mut mask_data = vec![0.0f32; total_len * total_len];

    let mut blank_offsets = Vec::new();
    let mut offset = context_len;
    for &len in blank_lens {
        blank_offsets.push(offset);
        offset += len;
    }

    for query in 0..total_len {
        for key in 0..total_len {
            let val = mask_value(query, key, context_len, blank_lens, &blank_offsets);
            mask_data[query * total_len + key] = val;
        }
    }

    Tensor::from_slice(&mask_data, (1, 1, total_len, total_len), device)
}

fn mask_value(
    query: usize,
    key: usize,
    context_len: usize,
    blank_lens: &[usize],
    _blank_offsets: &[usize],
) -> f32 {
    // Context token as query
    if query < context_len {
        // Context sees all: context + blanks
        return 0.0;
    }

    // Blank token as query: find which blank it belongs to
    let query_offset = query - context_len;
    let mut q_blank_idx = 0;
    let mut q_local_offset = 0;
    let mut accumulated = 0;

    for (i, &len) in blank_lens.iter().enumerate() {
        if query_offset < accumulated + len {
            q_blank_idx = i;
            q_local_offset = query_offset - accumulated;
            break;
        }
        accumulated += len;
    }

    // Key is in context
    if key < context_len {
        return 0.0;
    }

    // Key is in an earlier blank
    let key_offset = key - context_len;
    let mut k_blank_idx = 0;
    let mut k_accumulated = 0;

    for (i, &len) in blank_lens.iter().enumerate() {
        if key_offset < k_accumulated + len {
            k_blank_idx = i;
            break;
        }
        k_accumulated += len;
    }

    // Key is in an earlier blank: attend freely
    if k_blank_idx < q_blank_idx {
        return 0.0;
    }

    // Key is in a later blank: mask out
    if k_blank_idx > q_blank_idx {
        return f32::NEG_INFINITY;
    }

    // Key is in the same blank: causal (can see earlier and current)
    let k_local_offset = key_offset - k_accumulated;
    if k_local_offset <= q_local_offset {
        0.0
    } else {
        f32::NEG_INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_mask_no_blanks() -> Result<()> {
        let device = Device::Cpu;
        let mask = build_glm_mask(4, &[], &device)?;
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        // mask shape [1, 1, 4, 4]; extract batch=0, head=0 -> [4, 4]
        let m = mask.get(0)?.get(0)?.to_vec2::<f32>()?;
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m[i][j], 0.0, "m[{i}][{j}] should be 0");
            }
        }
        Ok(())
    }

    #[test]
    fn test_blank_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = build_glm_mask(2, &[3], &device)?;
        assert_eq!(mask.dims(), &[1, 1, 5, 5]);
        let m = mask.get(0)?.get(0)?.to_vec2::<f32>()?;
        // Context rows (0,1) see everything (0.0)
        assert_eq!(m[0][4], 0.0);
        assert_eq!(m[1][4], 0.0);
        // First blank token (row 2) sees context + itself = 0.0
        assert_eq!(m[2][0], 0.0);
        assert_eq!(m[2][2], 0.0);
        // First blank token should NOT see later blank tokens
        assert_eq!(m[2][4], f32::NEG_INFINITY);
        // Last blank token (row 4) sees all context + all blanks = 0.0
        assert_eq!(m[4][0], 0.0);
        assert_eq!(m[4][4], 0.0);
        Ok(())
    }
}


