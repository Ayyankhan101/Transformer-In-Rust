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
