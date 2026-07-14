use candle_core::{Device, Result, Tensor};

use crate::layers::attention::MultiHeadAttention;
use crate::layers::embedding::Embedding;
use crate::layers::ffn::{Activation, FeedForward};
use crate::layers::norm::LayerNorm;

use super::config::CodeGenConfig;
use super::kv_cache::KVCache;
use super::rotary::RotaryEmbedding;

pub struct CodeGenModel {
    pub embedding: Embedding,
    pub blocks: Vec<CodeGenBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Tensor,
    pub lm_head_bias: Tensor,
    pub rotary: RotaryEmbedding,
    config: CodeGenConfig,
}

pub struct CodeGenBlock {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub norm2: LayerNorm,
    pub ffn: FeedForward,
}

impl CodeGenBlock {
    pub fn new_blank(config: &CodeGenConfig, device: &Device) -> Result<Self> {
        let dtype = config.dtype;
        let norm1 = LayerNorm::zeros_with_dtype(config.hidden_dim, config.eps, dtype, device)?;
        let norm2 = LayerNorm::zeros_with_dtype(config.hidden_dim, config.eps, dtype, device)?;
        let attn = MultiHeadAttention::new_blank(config.hidden_dim, config.num_heads, device)?;
        let ffn = FeedForward::new_blank(Activation::GELU, config.hidden_dim, config.ffn_dim, device)?;
        Ok(Self { norm1, norm2, attn, ffn })
    }
}

impl CodeGenModel {
    pub fn new_blank(config: CodeGenConfig, device: &Device) -> Result<Self> {
        let dtype = config.dtype;
        let embedding = Embedding::zeros(config.vocab_size, config.hidden_dim, device)?;
        let rotary = RotaryEmbedding::new(config.rotary_dim, config.max_seq_len, dtype, device)?;

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(CodeGenBlock::new_blank(&config, device)?);
        }

        let final_norm = LayerNorm::zeros_with_dtype(config.hidden_dim, config.eps, dtype, device)?;
        let lm_head = Tensor::zeros((config.hidden_dim, config.vocab_size), dtype, device)?;
        let lm_head_bias = Tensor::zeros(config.vocab_size, dtype, device)?;

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            lm_head,
            lm_head_bias,
            rotary,
            config,
        })
    }

    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        positions: &[usize],
        cache: &mut Option<Vec<KVCache>>,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(token_ids)?;
        let max_seq = self.config.max_seq_len;
        let n_heads = self.config.num_heads;
        let head_dim = self.config.head_dim();

        if cache.is_none() {
            let device = x.device().clone();
            let mut caches = Vec::with_capacity(self.blocks.len());
            for _ in 0..self.blocks.len() {
                caches.push(KVCache::new(max_seq, n_heads, head_dim, &device)?);
            }
            *cache = Some(caches);
        }
        let caches = cache.as_mut().unwrap();

        for (i, block) in self.blocks.iter().enumerate() {
            let normed = block.norm1.forward(&x)?;
            let attn_out = {
                let qkv = normed.broadcast_matmul(&block.attn.qkv_weight().unsqueeze(0)?)?;
                let (bs, sl, _) = qkv.dims3()?;
                let qkv = qkv.reshape((bs, sl, 3, self.config.num_heads, head_dim))?;
                let qkv = qkv.permute((0, 3, 2, 1, 4))?;
                let q = qkv.get_on_dim(2, 0)?;
                let v = qkv.get_on_dim(2, 1)?;
                let k = qkv.get_on_dim(2, 2)?;

                let q_rot = self.rotary.apply_rotary(&q, positions)?;
                let k_rot = self.rotary.apply_rotary(&k, positions)?;

                let (k_full, v_full) = caches[i].append(&k_rot, &v)?;

                let scale = 1.0 / (head_dim as f64).sqrt();
                let mut scores = q_rot.broadcast_matmul(&k_full.transpose(2, 3)?)?;
                scores = (scores * scale)?;

                let (_, _, q_len, _) = q_rot.dims4()?;
                let (_, _, kv_len, _) = k_full.dims4()?;
                if q_len > 1 && q_len == kv_len {
                    let mask = crate::layers::attention::causal_mask(q_len, scores.device())?;
                    scores = scores.broadcast_add(&mask)?;
                }

                let weights = candle_nn::ops::softmax(&scores, 3)?;
                let context = weights.broadcast_matmul(&v_full)?;

                let context = context.permute((0, 2, 1, 3))?.reshape((bs, sl, self.config.hidden_dim))?;
                context.broadcast_matmul(&block.attn.out_weight().unsqueeze(0)?)
            };
            let attn_out = attn_out?;
            let ffn_out = block.ffn.forward(&normed)?;
            x = (x + attn_out + ffn_out)?;
        }

        let x = self.final_norm.forward(&x)?;
        let logits = x.broadcast_matmul(&self.lm_head.unsqueeze(0)?)?;
        logits.broadcast_add(&self.lm_head_bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_blank_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = CodeGenConfig::default();
        let model = CodeGenModel::new_blank(config, &device)?;
        let ids = &[1u32, 2, 3, 4, 5];
        let positions = &[0usize, 1, 2, 3, 4];
        let logits = model.forward_with_cache(ids, positions, &mut None)?;
        assert_eq!(logits.dims(), &[1, 5, 50400]);
        Ok(())
    }
}
