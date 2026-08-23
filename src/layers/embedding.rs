use candle_core::{DType, Device, Result, Tensor};

pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn zeros(
        vocab_size: usize,
        hidden_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let weight = Tensor::zeros((vocab_size, hidden_dim), dtype, device)?;
        Ok(Self { weight })
    }

    pub fn new(vocab_size: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        let weight = Tensor::randn(0.0f32, 1.0f32, (vocab_size, hidden_dim), device)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::from_slice(ids, ids.len(), self.weight.device())?;
        self.weight.index_select(&ids, 0)?.unsqueeze(0)
    }

    pub fn from_tensor(weight: Tensor) -> Self {
        Self { weight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_lookup() -> Result<()> {
        let device = Device::Cpu;
        let emb = Embedding::new(100, 32, &device)?;
        let ids = vec![5u32, 10, 0];
        let out = emb.forward(&ids)?;
        assert_eq!(out.dims(), &[1, 3, 32]);
        Ok(())
    }
}
