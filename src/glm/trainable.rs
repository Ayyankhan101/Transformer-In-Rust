use candle_core::{Device, Result, Tensor, Var};

use crate::glm::config::GLMConfig;
use crate::layers::attention::causal_mask;

// --- Trainable Embedding ---
pub struct TrainableEmbedding {
    pub weight: Var,
}

impl TrainableEmbedding {
    pub fn new(vocab_size: usize, hidden_dim: usize, device: &Device) -> Result<Self> {
        let t = Tensor::randn(0.0f32, 1.0f32, (vocab_size, hidden_dim), device)?;
        let weight = Var::from_tensor(&t)?;
        Ok(Self { weight })
    }

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let mut rows = Vec::new();
        for &id in ids {
            rows.push(self.weight.get(id as usize)?);
        }
        Tensor::stack(&rows, 0)?.unsqueeze(0)
    }
}

// --- Trainable RMSNorm ---
pub struct TrainableRMSNorm {
    weight: Var,
    eps: f64,
}

impl TrainableRMSNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Result<Self> {
        let t = Tensor::ones(dim, candle_core::DType::F32, device)?;
        let weight = Var::from_tensor(&t)?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let last_dim = x.dims().len() - 1;
        let variance = x.sqr()?.mean(last_dim)?;
        let norm_x = (variance + self.eps)?.sqrt()?;
        let norm_x_3d = norm_x.unsqueeze(last_dim)?;
        x.broadcast_div(&norm_x_3d)?.broadcast_mul(&self.weight.as_tensor())
    }
}

// --- Trainable SwiGLU FeedForward ---
pub struct TrainableFeedForward {
    up: Var,
    gate: Var,
    down: Var,
}

impl TrainableFeedForward {
    pub fn new(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let up_t = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), device)?;
        let gate_t = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), device)?;
        let down_t = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), device)?;
        Ok(Self {
            up: Var::from_tensor(&up_t)?,
            gate: Var::from_tensor(&gate_t)?,
            down: Var::from_tensor(&down_t)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let up = x.broadcast_matmul(&self.up.as_tensor().unsqueeze(0)?)?;
        let gate = x.broadcast_matmul(&self.gate.as_tensor().unsqueeze(0)?)?;
        let gate_sig = crate::layers::ffn::swiglu_gate(&gate)?;
        let hidden = (up * gate_sig)?;
        hidden.broadcast_matmul(&self.down.as_tensor().unsqueeze(0)?)
    }
}

// --- Trainable Multi-Head Attention ---
pub struct TrainableAttention {
    qkv_weight: Var,
    out_weight: Var,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl TrainableAttention {
    pub fn new(hidden_dim: usize, num_heads: usize, device: &Device) -> Result<Self> {
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let qkv_t = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), device)?;
        let out_t = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), device)?;
        Ok(Self {
            qkv_weight: Var::from_tensor(&qkv_t)?,
            out_weight: Var::from_tensor(&out_t)?,
            num_heads, head_dim, scale,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_dim) = x.dims3()?;
        let qkv = x.broadcast_matmul(&self.qkv_weight.as_tensor().unsqueeze(0)?)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((0, 3, 2, 1, 4))?;
        let q = qkv.get_on_dim(2, 0)?;
        let k = qkv.get_on_dim(2, 1)?;
        let v = qkv.get_on_dim(2, 2)?;

        let scores = q.broadcast_matmul(&k.transpose(2, 3)?)?;
        let scaled_scores = (scores * self.scale)?;

        let masked = if let Some(m) = mask {
            if m.dtype() != scaled_scores.dtype() {
                scaled_scores.broadcast_add(&m.to_dtype(scaled_scores.dtype())?)?
            } else {
                scaled_scores.broadcast_add(m)?
            }
        } else {
            scaled_scores
        };

        let attention_weights = candle_nn::ops::softmax(&masked, 3)?;
        let context = attention_weights.broadcast_matmul(&v)?;
        let context = context.permute((0, 2, 1, 3))?.reshape((batch_size, seq_len, _hidden_dim))?;
        context.broadcast_matmul(&self.out_weight.as_tensor().unsqueeze(0)?)
    }
}

// --- Trainable Transformer Block ---
pub struct TrainableBlock {
    norm1: TrainableRMSNorm,
    attn: TrainableAttention,
    norm2: TrainableRMSNorm,
    ffn: TrainableFeedForward,
}

impl TrainableBlock {
    pub fn new(hidden_dim: usize, num_heads: usize, ffn_dim: usize, eps: f64, device: &Device) -> Result<Self> {
        Ok(Self {
            norm1: TrainableRMSNorm::new(hidden_dim, eps, device)?,
            attn: TrainableAttention::new(hidden_dim, num_heads, device)?,
            norm2: TrainableRMSNorm::new(hidden_dim, eps, device)?,
            ffn: TrainableFeedForward::new(hidden_dim, ffn_dim, device)?,
        })
    }

    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed, mask)?;
        let x = x.broadcast_add(&attn_out)?;
        let normed = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed)?;
        x.broadcast_add(&ffn_out)
    }
}

// --- Full Trainable GLM Model ---
pub struct TrainableGLMModel {
    pub embedding: TrainableEmbedding,
    pub pos_1_embedding: Var,
    pub pos_2_embedding: Var,
    pub blocks: Vec<TrainableBlock>,
    pub final_norm: TrainableRMSNorm,
    pub lm_head: Var,
}

impl TrainableGLMModel {
    pub fn new(config: GLMConfig, device: &Device) -> Result<Self> {
        let embedding = TrainableEmbedding::new(config.vocab_size, config.hidden_dim, device)?;

        let p1 = Tensor::randn(0.0f32, 0.02f32, (config.max_seq_len, config.hidden_dim), device)?;
        let p2 = Tensor::randn(0.0f32, 0.02f32, (config.max_seq_len, config.hidden_dim), device)?;

        let mut blocks = Vec::new();
        for _ in 0..config.num_layers {
            blocks.push(TrainableBlock::new(
                config.hidden_dim, config.num_heads,
                config.ffn_dim, config.eps, device,
            )?);
        }

        let final_norm = TrainableRMSNorm::new(config.hidden_dim, config.eps, device)?;
        let lm_t = Tensor::randn(0.0f32, 0.02f32, (config.hidden_dim, config.vocab_size), device)?;

        Ok(Self {
            embedding,
            pos_1_embedding: Var::from_tensor(&p1)?,
            pos_2_embedding: Var::from_tensor(&p2)?,
            blocks, final_norm,
            lm_head: Var::from_tensor(&lm_t)?,
        })
    }

    fn apply_positions(&self, x: &Tensor, context_len: usize, blank_lens: &[usize]) -> Result<Tensor> {
        let seq_len = x.dims()[1];
        if context_len == 0 && blank_lens.is_empty() {
            let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
            let pos_tensor = Tensor::new(pos_ids.as_slice(), x.device())?;
            let pos_emb = self.pos_1_embedding.index_select(&pos_tensor, 0)?;
            x.broadcast_add(&pos_emb.unsqueeze(0)?)
        } else {
            let total_len = context_len + blank_lens.iter().sum::<usize>();
            let mut pos_1_ids = Vec::new();
            let mut pos_2_ids = Vec::new();

            for i in 0..context_len {
                pos_1_ids.push(0u32);
                pos_2_ids.push(i as u32);
            }

            for (blank_idx, &blank_len) in blank_lens.iter().enumerate() {
                for local_offset in 0..blank_len {
                    pos_1_ids.push((blank_idx + 1) as u32);
                    pos_2_ids.push(local_offset as u32);
                }
            }

            let pos_1_tensor = Tensor::from_slice(&pos_1_ids, total_len, x.device())?;
            let pos_2_tensor = Tensor::from_slice(&pos_2_ids, total_len, x.device())?;

            let pos_1_emb = self.pos_1_embedding.index_select(&pos_1_tensor, 0)?;
            let pos_2_emb = self.pos_2_embedding.index_select(&pos_2_tensor, 0)?;

            let combined = pos_1_emb.broadcast_add(&pos_2_emb)?;
            x.broadcast_add(&combined.unsqueeze(0)?)
        }
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        context_len: usize,
        blank_lens: &[usize],
        mask: &Tensor,
    ) -> Result<Tensor> {
        let mut x = self.embedding.forward(token_ids)?;
        x = self.apply_positions(&x, context_len, blank_lens)?;

        for block in &self.blocks {
            x = block.forward(&x, Some(mask))?;
        }

        let x = self.final_norm.forward(&x)?;
        x.broadcast_matmul(&self.lm_head.as_tensor().unsqueeze(0)?)
    }

    pub fn forward_causal(&self, token_ids: &[u32]) -> Result<Tensor> {
        let seq_len = token_ids.len();
        let mask = causal_mask(seq_len, &self.lm_head.device())?;
        self.forward(token_ids, 0, &[], &mask)
    }

    pub fn param_vars(&self) -> Vec<Var> {
        let mut params = Vec::new();
        params.push(self.embedding.weight.clone());
        params.push(self.pos_1_embedding.clone());
        params.push(self.pos_2_embedding.clone());
        for block in &self.blocks {
            params.push(block.norm1.weight.clone());
            params.push(block.attn.qkv_weight.clone());
            params.push(block.attn.out_weight.clone());
            params.push(block.norm2.weight.clone());
            params.push(block.ffn.up.clone());
            params.push(block.ffn.gate.clone());
            params.push(block.ffn.down.clone());
        }
        params.push(self.final_norm.weight.clone());
        params.push(self.lm_head.clone());
        params
    }
}
