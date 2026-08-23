use candle_core::{DType, Result, Tensor};
use rand::Rng;

/// Sample one token from `logits`.
///
/// `rng` is supplied by the caller so a whole generation can share one stream —
/// re-seeding per token would draw the same value every step.
pub fn sample(
    logits: &Tensor,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    repetition_penalty: f64,
    seen_tokens: &[u32],
    rng: &mut impl Rng,
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

    let r: f32 = rng.gen();

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
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_argmax_sampling() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 100];
        data[42] = 10.0;
        let logits = Tensor::from_vec(data, 100, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 0.0, 1, 1.0, 1.0, &[], &mut rng)?;
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
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 0.0, 50, 1.0, 1.0, &[], &mut rng)?;
        assert_eq!(token, 99);
        Ok(())
    }

    #[test]
    fn test_repetition_penalty() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![1.0f32; 10];
        let logits = Tensor::from_vec(data, 10, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 0.0, 10, 1.0, 2.0, &[5], &mut rng)?;
        assert_ne!(token, 5);
        Ok(())
    }

    #[test]
    fn test_top_k_filters_low_tokens() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 100];
        data[0] = 10.0;
        data[1] = 9.0;
        data[2] = 8.0;
        let logits = Tensor::from_vec(data, 100, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 0.0, 3, 1.0, 1.0, &[], &mut rng)?;
        assert!(token <= 2);
        Ok(())
    }

    #[test]
    fn test_top_p_filters_tail() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![0.0f32; 100];
        data[0] = 10.0;
        data[1] = 9.0;
        let logits = Tensor::from_vec(data, 100, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 1.0, 100, 0.5, 1.0, &[], &mut rng)?;
        assert!(token <= 1);
        Ok(())
    }

    #[test]
    fn test_uniform_distribution() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![1.0f32; 10];
        let logits = Tensor::from_vec(data, 10, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 1.0, 10, 1.0, 1.0, &[], &mut rng)?;
        assert!(token < 10);
        Ok(())
    }

    #[test]
    fn test_single_token_vocab() -> Result<()> {
        let device = Device::Cpu;
        let data = vec![5.0f32];
        let logits = Tensor::from_vec(data, 1, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 1.0, 1, 1.0, 1.0, &[], &mut rng)?;
        assert_eq!(token, 0);
        Ok(())
    }

    #[test]
    fn test_negative_logits() -> Result<()> {
        let device = Device::Cpu;
        let mut data = vec![-10.0f32; 50];
        data[25] = -1.0;
        let logits = Tensor::from_vec(data, 50, &device)?;
        let mut rng = StdRng::seed_from_u64(7);
        let token = sample(&logits, 0.0, 50, 1.0, 1.0, &[], &mut rng)?;
        assert_eq!(token, 25);
        Ok(())
    }

    /// Regression: sampling used to fall through to argmax for every token
    /// because the RNG produced values far outside [0, 1).
    #[test]
    fn test_sampling_is_not_always_argmax() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32; 32], 32, &device)?;
        let mut rng = StdRng::seed_from_u64(7);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..50 {
            seen.insert(sample(&logits, 1.0, 32, 1.0, 1.0, &[], &mut rng)?);
        }
        assert!(
            seen.len() > 1,
            "uniform logits should not always yield the same token, got {seen:?}"
        );
        Ok(())
    }

    /// Same seed, same sequence — reproducibility for `--seed`.
    #[test]
    fn test_sampling_is_reproducible_for_a_given_seed() -> Result<()> {
        let device = Device::Cpu;
        let logits = Tensor::from_vec(vec![0.0f32; 32], 32, &device)?;

        let draw = |seed: u64| -> Result<Vec<u32>> {
            let mut rng = StdRng::seed_from_u64(seed);
            (0..10)
                .map(|_| sample(&logits, 1.0, 32, 1.0, 1.0, &[], &mut rng))
                .collect()
        };

        assert_eq!(draw(11)?, draw(11)?);
        assert_ne!(draw(11)?, draw(12)?);
        Ok(())
    }
}
