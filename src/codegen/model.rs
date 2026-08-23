use candle_core::{Device, Result, Tensor};

use crate::layers::attention::MultiHeadAttention;
use crate::layers::embedding::Embedding;
use crate::layers::ffn::{Activation, FeedForward};
use crate::layers::norm::LayerNorm;

use super::config::CodeGenConfig;
use super::kv_cache::KVCache;
use super::rotary::RotaryEmbedding;

/// Number of model-parallel shards baked into CodeGen's fused `qkv_proj` weight.
///
/// The projection is not stored as `[all q | all v | all k]`. It keeps the
/// sharded layout of the original TPU implementation: `MP_NUM` consecutive
/// groups, each holding `[q | v | k]` for its slice of the heads. Mirrors
/// `mp_num = 4` in HuggingFace `models/codegen/modeling_codegen.py`.
const MP_NUM: usize = 4;

/// Split CodeGen's fused QKV projection into `[batch, heads, seq, head_dim]`
/// tensors, undoing the model-parallel group interleaving described on
/// [`MP_NUM`].
fn split_qkv(qkv: &Tensor, num_heads: usize, head_dim: usize) -> Result<(Tensor, Tensor, Tensor)> {
    if num_heads % MP_NUM != 0 {
        return Err(candle_core::Error::Msg(format!(
            "num_heads ({num_heads}) must be divisible by the CodeGen model-parallel group count ({MP_NUM})"
        )));
    }
    let (bs, sl, _) = qkv.dims3()?;
    let local_dim = num_heads * head_dim / MP_NUM;
    // [bs, sl, MP_NUM, 3, local_dim] — dim 3 selects q, v, k in that order.
    let grouped = qkv.reshape((bs, sl, MP_NUM, 3, local_dim))?;

    let mut split = Vec::with_capacity(3);
    for slot in 0..3 {
        // Flattening the group and per-group-head axes yields head index
        // `group * (num_heads / MP_NUM) + i`, matching `_split_heads` upstream.
        let part = grouped
            .get_on_dim(3, slot)?
            .contiguous()?
            .reshape((bs, sl, num_heads, head_dim))?;
        split.push(part.permute((0, 2, 1, 3))?.contiguous()?);
    }
    let k = split.pop().unwrap();
    let v = split.pop().unwrap();
    let q = split.pop().unwrap();
    Ok((q, k, v))
}

pub struct CodeGenModel {
    pub embedding: Embedding,
    pub blocks: Vec<CodeGenBlock>,
    pub final_norm: LayerNorm,
    pub lm_head: Tensor,
    pub lm_head_bias: Tensor,
    pub rotary: RotaryEmbedding,
    config: CodeGenConfig,
}

/// A CodeGen transformer block.
///
/// Attention and FFN are *parallel* (GPT-J style): both read the same
/// `norm1` output and their results are summed into one residual, so unlike a
/// serial pre-norm block there is only one LayerNorm per block.
pub struct CodeGenBlock {
    pub norm1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ffn: FeedForward,
}

impl CodeGenBlock {
    pub fn new_blank(config: &CodeGenConfig, device: &Device) -> Result<Self> {
        let dtype = config.dtype;
        let norm1 = LayerNorm::zeros_with_dtype(config.hidden_dim, config.eps, dtype, device)?;
        let attn =
            MultiHeadAttention::new_blank(config.hidden_dim, config.num_heads, dtype, device)?;
        let ffn = FeedForward::new_blank(
            Activation::GELU,
            config.hidden_dim,
            config.ffn_dim,
            dtype,
            device,
        )?;
        Ok(Self { norm1, attn, ffn })
    }
}

impl CodeGenModel {
    pub fn new_blank(config: CodeGenConfig, device: &Device) -> Result<Self> {
        let dtype = config.dtype;
        let embedding = Embedding::zeros(config.vocab_size, config.hidden_dim, dtype, device)?;
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

    /// Full forward pass, logits for every input position.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        positions: &[usize],
        cache: &mut Option<Vec<KVCache>>,
    ) -> Result<Tensor> {
        let hidden = self.forward_hidden(token_ids, positions, cache)?;
        self.project_logits(&hidden)
    }

    /// Everything up to and including the final norm, without the vocabulary
    /// projection.
    ///
    /// Generation only ever reads the last position, so it calls this, narrows,
    /// and projects one row — the `lm_head` matmul is `hidden_dim × vocab_size`
    /// per position and would otherwise be paid for the whole prompt at prefill.
    pub fn forward_hidden(
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
            let dtype = self.config.dtype;
            let mut caches = Vec::with_capacity(self.blocks.len());
            for _ in 0..self.blocks.len() {
                caches.push(KVCache::new(max_seq, n_heads, head_dim, dtype, &device)?);
            }
            *cache = Some(caches);
        }
        let caches = cache.as_mut().unwrap();

        for (i, block) in self.blocks.iter().enumerate() {
            let normed = block.norm1.forward(&x)?;
            let attn_out = {
                let qkv = normed.broadcast_matmul(&block.attn.qkv_weight().unsqueeze(0)?)?;
                let (bs, sl, _) = qkv.dims3()?;
                let (q, k, v) = split_qkv(&qkv, self.config.num_heads, head_dim)?;

                let q_rot = self.rotary.apply_rotary(&q, positions)?;
                let k_rot = self.rotary.apply_rotary(&k, positions)?;

                let (k_full, v_full) = caches[i].append(&k_rot, &v)?;

                let scale = 1.0 / (head_dim as f64).sqrt();
                let mut scores = q_rot.broadcast_matmul(&k_full.transpose(2, 3)?)?;
                scores = (scores * scale)?;

                let (_, _, q_len, _) = q_rot.dims4()?;
                let (_, _, kv_len, _) = k_full.dims4()?;
                if q_len > 1 && q_len == kv_len {
                    let mask = crate::layers::attention::causal_mask(
                        q_len,
                        scores.device(),
                        scores.dtype(),
                    )?;
                    scores = scores.broadcast_add(&mask)?;
                }

                let weights = candle_nn::ops::softmax(&scores, 3)?;
                let context = weights.broadcast_matmul(&v_full)?;

                let context =
                    context
                        .permute((0, 2, 1, 3))?
                        .reshape((bs, sl, self.config.hidden_dim))?;
                context.broadcast_matmul(&block.attn.out_weight().unsqueeze(0)?)
            };
            let attn_out = attn_out?;
            let ffn_out = block.ffn.forward(&normed)?;
            x = (x + attn_out + ffn_out)?;
        }

        self.final_norm.forward(&x)
    }

    /// Project hidden states onto the vocabulary.
    pub fn project_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        let logits = hidden.broadcast_matmul(&self.lm_head.unsqueeze(0)?)?;
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

    /// Regression: `new_blank` hardcoded F32 weights regardless of `config.dtype`,
    /// so an F16 model panicked with "dtype mismatch in mul" on the first forward.
    #[test]
    fn test_codegen_blank_forward_f16() -> Result<()> {
        let device = Device::Cpu;
        let config = CodeGenConfig {
            dtype: candle_core::DType::F16,
            vocab_size: 64,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            ffn_dim: 64,
            max_seq_len: 16,
            rotary_dim: 8,
            ..Default::default()
        };
        let model = CodeGenModel::new_blank(config, &device)?;
        let logits = model.forward_with_cache(&[1u32, 2, 3], &[0usize, 1, 2], &mut None)?;
        assert_eq!(logits.dims(), &[1, 3, 64]);
        assert_eq!(logits.dtype(), candle_core::DType::F16);
        Ok(())
    }
}
