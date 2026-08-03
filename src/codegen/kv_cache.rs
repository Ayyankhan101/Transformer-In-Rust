use candle_core::{DType, Device, Result, Tensor};

pub struct KVCache {
    k: Tensor,
    v: Tensor,
    pos: usize,
    n_heads: usize,
    head_dim: usize,
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
            n_heads,
            head_dim,
        })
    }

    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = k_new.dim(2)?;
        let end = self.pos + seq_len;

        let k = self.k.slice_assign(
            &[0..1, 0..self.n_heads, self.pos..end, 0..self.head_dim],
            k_new,
        )?;
        let v = self.v.slice_assign(
            &[0..1, 0..self.n_heads, self.pos..end, 0..self.head_dim],
            v_new,
        )?;

        self.k = k;
        self.v = v;
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
    fn test_kv_cache_dtype_f16() {
        let device = Device::Cpu;
        let cache = KVCache::new(64, 4, 32, DType::F16, &device).unwrap();
        assert_eq!(cache.position(), 0);
    }
}
