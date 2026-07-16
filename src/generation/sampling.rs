use candle_core::{DType, Result, Tensor};

pub fn sample(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
    seen_tokens: &[u32],
    seed: u64,
) -> Result<u32> {
    let logits = if logits.dtype() != DType::F32 {
        logits.to_dtype(DType::F32)?
    } else {
        logits.clone()
    };

    if temperature <= 0.0 {
        return logits.argmax(0)?.to_scalar::<u32>();
    }

    let mut logits_vec = logits.to_vec1::<f32>()?;

    if (repetition_penalty - 1.0).abs() > f64::EPSILON {
        for &tid in seen_tokens {
            let idx = tid as usize;
            if idx < logits_vec.len() {
                if logits_vec[idx] < 0.0 {
                    logits_vec[idx] *= repetition_penalty as f32;
                } else {
                    logits_vec[idx] /= repetition_penalty as f32;
                }
            }
        }
    }

    let t = temperature as f32;
    for v in logits_vec.iter_mut() {
        *v /= t;
    }

    let vocab_size = logits_vec.len();
    let k = top_k.min(vocab_size);
    if k < vocab_size {
        let mut indexed: Vec<(usize, f32)> = logits_vec
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.select_nth_unstable_by(vocab_size - k, |a, b| a.1.partial_cmp(&b.1).unwrap());
        let threshold = indexed[vocab_size - k].1;
        for v in logits_vec.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    let max_logit = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in logits_vec.iter_mut() {
        *v = (*v - max_logit).exp();
        sum += *v;
    }
    for v in logits_vec.iter_mut() {
        *v /= sum;
    }

    let mut indexed: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut cumulative = 0.0f32;
    let mut cutoff = indexed.len();
    for (j, &(_, p)) in indexed.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p as f32 {
            cutoff = j + 1;
            break;
        }
    }
    indexed.truncate(cutoff);

    let sum: f32 = indexed.iter().map(|(_, p)| p).sum();
    for (_, p) in indexed.iter_mut() {
        *p /= sum;
    }

    let mut state = seed;
    let r = {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state as f32) / (u32::MAX as f32)
    };

    cumulative = 0.0;
    for &(idx, p) in &indexed {
        cumulative += p;
        if r <= cumulative {
            return Ok(idx as u32);
        }
    }

    Ok(indexed.first().map(|(i, _)| *i as u32).unwrap_or(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_argmax_sampling() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 100];
        data[42] = 10.0;
        let logits = Tensor::from_vec(data, 100, &device)?;
        let token = sample(&logits, 0.0, 1, 1.0, 1.0, &[], 0)?;
        assert_eq!(token, 42);
        Ok(())
    }

    #[test]
    fn test_temperature_zero_argmax() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 100];
        data[99] = 5.0;
        data[50] = 3.0;
        let logits = Tensor::from_vec(data, 100, &device)?;
        let token = sample(&logits, 0.0, 50, 1.0, 1.0, &[], 0)?;
        assert_eq!(token, 99);
        Ok(())
    }
}
