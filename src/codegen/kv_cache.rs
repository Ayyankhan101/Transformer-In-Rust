use candle_core::{DType, Device, Result, Tensor};

pub struct KVCache {
    k: Tensor,
    v: Tensor,
    pos: usize,
    max_seq_len: usize,
}

impl KVCache {
    pub fn new(
        max_seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let k = Tensor::zeros((1, n_heads, max_seq_len, head_dim), dtype, device)?;
        let v = Tensor::zeros((1, n_heads, max_seq_len, head_dim), dtype, device)?;
        Ok(Self {
            k,
            v,
            pos: 0,
            max_seq_len,
        })
    }

    /// Write `k_new`/`v_new` at the current position and return views over the
    /// cache up to and including them.
    ///
    /// Uses [`Tensor::slice_set`], which copies only the new tokens into the
    /// existing buffer. The previous `slice_assign` was not a targeted write: it
    /// zero-padded the source out to the full buffer shape, built a full-size
    /// mask, and ran `where_cond` over every element — the whole
    /// `[1, heads, max_seq_len, head_dim]` cache, twice per layer per token.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = k_new.dim(2)?;
        let end = self.pos + seq_len;
        if end > self.max_seq_len {
            return Err(candle_core::Error::Msg(format!(
                "KV cache overflow: {seq_len} more token(s) at position {} exceeds max_seq_len {}",
                self.pos, self.max_seq_len
            )));
        }

        // slice_set requires both sides contiguous; the rotary output may not be.
        self.k.slice_set(&k_new.contiguous()?, 2, self.pos)?;
        self.v.slice_set(&v_new.contiguous()?, 2, self.pos)?;
        self.pos = end;

        let k_out = self.k.narrow(2, 0, self.pos)?;
        let v_out = self.v.narrow(2, 0, self.pos)?;
        Ok((k_out, v_out))
    }

    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    #[allow(dead_code)]
    pub fn position(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_new() {
        let device = Device::Cpu;
        let cache = KVCache::new(128, 8, 64, DType::F32, &device).unwrap();
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn test_kv_cache_append_single() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(128, 2, 4, DType::F32, &device).unwrap();

        // Append 1 token
        let k_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 1, 4), &device).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 1, 4), &device).unwrap();

        let (k, v) = cache.append(&k_new, &v_new).unwrap();
        assert_eq!(cache.position(), 1);
        assert_eq!(k.dims(), &[1, 2, 1, 4]);
        assert_eq!(v.dims(), &[1, 2, 1, 4]);
    }

    #[test]
    fn test_kv_cache_append_multiple() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(128, 2, 4, DType::F32, &device).unwrap();

        // Append 3 tokens
        let k_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 3, 4), &device).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 3, 4), &device).unwrap();

        let (k, v) = cache.append(&k_new, &v_new).unwrap();
        assert_eq!(cache.position(), 3);
        assert_eq!(k.dims(), &[1, 2, 3, 4]);
        assert_eq!(v.dims(), &[1, 2, 3, 4]);

        // Append 2 more tokens
        let k_new2 = Tensor::randn(0.0f32, 1.0f32, (1, 2, 2, 4), &device).unwrap();
        let v_new2 = Tensor::randn(0.0f32, 1.0f32, (1, 2, 2, 4), &device).unwrap();

        let (k2, v2) = cache.append(&k_new2, &v_new2).unwrap();
        assert_eq!(cache.position(), 5);
        assert_eq!(k2.dims(), &[1, 2, 5, 4]);
        assert_eq!(v2.dims(), &[1, 2, 5, 4]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(128, 2, 4, DType::F32, &device).unwrap();

        let k_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 3, 4), &device).unwrap();
        let v_new = Tensor::randn(0.0f32, 1.0f32, (1, 2, 3, 4), &device).unwrap();

        cache.append(&k_new, &v_new).unwrap();
        assert_eq!(cache.position(), 3);

        cache.reset();
        assert_eq!(cache.position(), 0);
    }

    #[test]
    fn test_kv_cache_incremental_matches_block() {
        let device = Device::Cpu;
        let k = Tensor::randn(0.0f32, 1.0f32, (1, 2, 5, 4), &device).unwrap();
        let v = Tensor::randn(0.0f32, 1.0f32, (1, 2, 5, 4), &device).unwrap();

        let mut block = KVCache::new(16, 2, 4, DType::F32, &device).unwrap();
        let (k_block, v_block) = block.append(&k, &v).unwrap();

        let mut incremental = KVCache::new(16, 2, 4, DType::F32, &device).unwrap();
        let mut last = None;
        for i in 0..5 {
            let k_i = k.narrow(2, i, 1).unwrap();
            let v_i = v.narrow(2, i, 1).unwrap();
            last = Some(incremental.append(&k_i, &v_i).unwrap());
        }
        let (k_inc, v_inc) = last.unwrap();

        assert_eq!(k_inc.dims(), k_block.dims());
        for (a, b) in [(&k_inc, &k_block), (&v_inc, &v_block)] {
            let diff = (a - b)
                .unwrap()
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(diff < 1e-6, "incremental append diverged by {diff}");
        }
    }

    #[test]
    fn test_kv_cache_rejects_overflow() {
        let device = Device::Cpu;
        let mut cache = KVCache::new(4, 2, 4, DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 2, 5, 4), DType::F32, &device).unwrap();
        let v = Tensor::zeros((1, 2, 5, 4), DType::F32, &device).unwrap();
        let err = cache.append(&k, &v).unwrap_err().to_string();
        assert!(err.contains("KV cache overflow"), "unexpected error: {err}");
    }

    #[test]
    fn test_kv_cache_dtype_f16() {
        let device = Device::Cpu;
        let cache = KVCache::new(64, 4, 32, DType::F16, &device).unwrap();
        assert_eq!(cache.position(), 0);
    }
}
