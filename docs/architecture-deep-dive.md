# Architecture Deep-Dive

A technical walkthrough of the transformer architectures, layer implementations,
and design decisions in this project.

---

## Table of Contents

1. [Layer Library](#1-layer-library)
2. [GLM: Blank-Infilling Architecture](#2-glm-blank-infilling-architecture)
3. [CodeGen-350M: Causal Code Generation](#3-codegen-350m-causal-code-generation)
4. [Training Pipeline](#4-training-pipeline)
5. [Performance Optimization](#5-performance-optimization)
6. [FP16 Inference](#6-fp16-inference)
7. [Model Serialization](#7-model-serialization)

---

## 1. Layer Library

All shared building blocks live in `src/layers/`. Each layer is handwritten
using raw tensor operations from `candle_core`.

### 1.1 Embedding (`embedding.rs`)

A simple token embedding lookup table:

```rust
pub struct Embedding {
    weight: Tensor,  // shape: [vocab_size, hidden_dim]
}

impl Embedding {
    pub fn forward(&self, token_ids: &[u32]) -> Result<Tensor> {
        // Index into weight along dimension 0
        Tensor::index_select(&self.weight, 0, token_ids)
    }
}
```

**Design choice**: No learned positional embeddings here — GLM uses 2D learned
positions, CodeGen uses RoPE. Positional info is added outside the embedding layer.

### 1.2 Multi-Head Attention (`attention.rs`)

Standard scaled dot-product attention with fused QKV projection:

```
Input → QKV projection (fused) → split heads → scaled dot-product → out projection
```

Key details:
- **Fused QKV**: Single `weight` tensor of shape `[hidden_dim, 3 * hidden_dim]`
- **Causal mask**: Upper-triangular mask with -inf in the upper triangle
- **No bias**: Following GPT-2/CodeGen convention (LayerNorm provides bias)

**QKV layout**: CodeGen does *not* store the fused projection as
`[all q | all v | all k]`. It keeps the sharded layout of the original TPU
implementation — `mp_num = 4` consecutive groups, each holding `[q | v | k]` for
its slice of the heads. Splitting it any other way gives every head the wrong
slice of the projection, which is silent: shapes still line up, output is
garbage. See `split_qkv` in `src/codegen/model.rs`.

```rust
// [bs, sl, MP_NUM, 3, local_dim] — dim 3 selects q, v, k in that order
let grouped = qkv.reshape((bs, sl, MP_NUM, 3, local_dim))?;
let q = grouped.get_on_dim(3, 0)?.reshape((bs, sl, num_heads, head_dim))?;
// flattening group and per-group-head axes yields head `group * (heads / MP_NUM) + i`
```

**Causal mask optimization**: Instead of creating a full `[seq, seq]` mask each
forward pass, we create it once per call and cache it at the call site.

### 1.3 Feed-Forward Network (`ffn.rs`)

Two activation variants:

**SwiGLU** (used by GLM):
```
SwiGLU(x) = (x @ W_gate) ⊗ silu(x @ W_up) @ W_down
```
- Three weight matrices: up, gate, down
- Gated design provides better quality per parameter

**GELU** (used by CodeGen):
```
GELU(x) = x @ W_fc_in → GELU activation → x @ W_fc_out
```
- Two weight matrices: fc_in, fc_out
- Same design as GPT-2/CodeGen original

### 1.4 Layer Normalization (`norm.rs`)

Two variants:

**LayerNorm**: with learnable weight (γ) and bias (β)
```
y = γ * (x - μ) / √(σ² + ε) + β
```

**RMSNorm**: no bias, no mean subtraction (faster)
```
y = γ * x / √(x̄² + ε)
```

RMSNorm is used in GLM; LayerNorm in CodeGen (matching original implementations).

### 1.5 Transformer Block (`block.rs`)

Two patterns:

**Pre-Norm (GLM)**: Used in most modern transformers
```
x = x + MHA(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

**Parallel Sub-Blocks (CodeGen)**: GPT-J style, attention and FFN in parallel
```
x = x + MHA(LayerNorm(x)) + FFN(LayerNorm(x))
```

The parallel pattern is more efficient: one norm call instead of two.

---

## 2. GLM: Blank-Infilling Architecture

### 2.1 Overview

GLM (General Language Model) trains on span corruption: randomly mask spans of
tokens and predict the masked tokens using bidirectional context.

### 2.2 2D Positional Encoding

Unlike standard 1D positions, GLM uses two position vectors:
- **pos_1**: Position within the original sequence (all tokens, including masked)
- **pos_2**: Position within the current block (0 for context, 1, 2, 3... for masked tokens)

This allows the model to distinguish:
1. Where a token is in the overall sequence
2. Whether a token is context (pos_2 = 0) or part of a blank to fill (pos_2 > 0)

```rust
// In forward pass:
let pos_1_emb = self.pos_1.embed(pos_1_ids)?; // shape: [seq, hidden_dim]
let pos_2_emb = self.pos_2.embed(pos_2_ids)?; // shape: [seq, hidden_dim]
let x = tok_emb + pos_1_emb + pos_2_emb;
```

### 2.3 Blank-Infilling Mask

The attention mask handles three types of token relationships:
1. **Context → Context**: Full bidirectional attention
2. **Context → Blank**: Only left-to-right within a blank span
3. **Blank → Context**: Full attention (blank can see all context)
4. **Blank → Blank (different spans)**: No attention (spans are independent)
5. **Blank → Blank (same span)**: Left-to-right causal within the span

This creates a block-sparse attention pattern:

```
     C C C C B1 B1 B2 B2 B2
C    [1 1 1 1 1 1 1 1 1 1]
C    [1 1 1 1 1 1 1 1 1 1]
C    [1 1 1 1 1 1 1 1 1 1]
C    [1 1 1 1 1 1 1 1 1 1]
B1   [1 1 1 1 1 0 0 0 0 0]
B1   [1 1 1 1 1 1 0 0 0 0]
B2   [1 1 1 1 0 0 1 0 0 0]
B2   [1 1 1 1 0 0 1 1 0 0]
B2   [1 1 1 1 0 0 1 1 1 1]
```

Where C = context, B1 = blank span 1, B2 = blank span 2.

### 2.4 Training Objective

Loss is computed **only on corrupted positions** (the blank tokens), not on
context tokens. This forces the model to learn infilling rather than just
language modeling.

### 2.5 Training Forward

```rust
fn forward_causal(&self, ...) -> Result<Tensor> {
    let x = self.embedding(token_ids) + pos_1 + pos_2;
    for block in &self.blocks {
        x = x + block.attn(block.norm1(x), mask);
        x = x + block.ffn(block.norm2(x));
    }
    let logits = self.lm_head(self.final_norm(x));
    // Loss computed only on corrupted positions
    let loss = cross_entropy(logits.masked_select(corrupted_positions), targets);
    Ok(loss)
}
```

---

## 3. CodeGen-350M: Causal Code Generation

### 3.1 Architecture

Salesforce's CodeGen-350M is a GPT-2-style decoder-only model with:

- **20 transformer layers** with hidden_dim=1024, 16 heads (head_dim=64)
- **Rotary Position Embedding (RoPE)** instead of learned positions
- **KV Cache** for efficient autoregressive generation
- **GELU activation** in feed-forward networks
- **Parallel sub-blocks** (attention + FFN on same normalized input)

### 3.2 RoPE Implementation

RoPE applies a rotation to query and key vectors based on position, avoiding
learned positional embeddings:

```
For each pair (2i, 2i+1) in head_dim:
    θ = position / base^{2i / head_dim}
    q_rot[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
    q_rot[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
```

Key properties:
- **Relative attention**: The dot product Q·K only depends on relative position
- **Decay with distance**: As positions get further apart, attention decays
- No additional parameters needed (unlike learned positions)

### 3.3 KV Cache

During autoregressive generation, we cache K and V activations from previous
steps to avoid recomputation:

```
Prefill:  compute K, V for all prompt tokens → store in cache
Step 1:   compute K, V for token 1 → append to cache → attend to all cached
Step 2:   compute K, V for token 2 → append to cache → attend to all cached
...
```

This reduces the per-token computation from O(seq²) to O(seq).

### 3.4 Weight Loading

Weights are loaded from HuggingFace PyTorch checkpoints:

```rust
// WeightLoader::load picks the format from the extension
WeightLoader::load(path, &config, &device)?;      // .safetensors or .bin
```

`WeightLoader` handles name mapping and dtype conversion, with zero-init to avoid
allocating 350M random floats. A tensor named in the model but absent from the
checkpoint is an error, not a silently zeroed layer. `ModelContext` prefers
`model.safetensors` over `pytorch_model.bin` when both are present.

### 3.5 Sampling Pipeline

Generation uses a multi-stage sampling pipeline:

```
Logits → Repetition Penalty → Temperature → Top-K → Top-P → Sample
```

1. **Repetition Penalty**: Penalize already-seen tokens (1.0 = no penalty)
2. **Temperature**: Scale logits (0 = argmax, 1.0 = standard sampling)
3. **Top-K**: Keep only top K logits (40 = default)
4. **Top-P**: Keep smallest set of tokens whose cumulative prob ≥ 0.9
5. **Random Sample**: Sample from remaining distribution

---

## 4. Training Pipeline

### 4.1 Data Loading

```rust
pub struct DataLoader {
    train_examples: Vec<TrainingExample>,
    eval_examples: Vec<TrainingExample>,
    batch_size: usize,
    max_seq_len: usize,
}
```

- Reads text files from `data_dir/train/` and `data_dir/eval/`
- Tokenizes with BPE tokenizer
- Extracts random windows of `max_seq_len` tokens
- Shuffles at epoch boundaries

### 4.2 LR Scheduler

Three schedules available:

| Schedule | Formula | Best For |
|----------|---------|----------|
| Cosine | η = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * step / total)) | Most training runs |
| Linear | η = η_max * (1 - step / total) | Fine-tuning |
| Constant | η = η_max | Debugging |

All support linear warmup:
```rust
if step < warmup_steps {
    lr = lr * (step as f64 / warmup_steps as f64);
}
```

### 4.3 Gradient Clipping

Global norm clipping prevents gradient explosion:
```rust
let grad_norm = params.iter()
    .map(|p| p.grad().unwrap().sqr().sum_all())
    .sum::<Tensor>()?.sqrt()?;
let scale = (max_grad_norm / grad_norm).min(1.0);
for p in params { *p.grad() *= scale; }
```

### 4.4 Checkpoint System

Checkpoints contain:
- **model_step_NNNNNNN.safetensors**: Full model weights in HF-compatible format
- **optimizer_step_NNNNNNN.json**: Optimizer state (step, learning rate, momentum)
- **training_state.json**: Current step, best loss, configuration

Auto-cleanup keeps only the N most recent checkpoints.

---

## 5. Performance Optimization

### 5.1 FP16 Inference

All weights and activations can use F16 (half precision):
- **Memory**: ~50% reduction (700MB → 350MB for CodeGen)
- **Speed**: ~23% faster on i5-6600 with `gemm` F16 support
- **Quality**: Negligible degradation for inference

```rust
// Cargo.toml
gemm = { version = "0.17", features = ["f16"] }
```

### 5.2 Matrix Multiplication

Candle uses `gemm` crate for CPU matmul, which provides:
- F32: Standard GEMM
- F16: Half-precision GEMM via gemm's f16 feature
- ARM NEON: Auto-detected on aarch64

### 5.3 Memory Layout

- Weights stored in contiguous row-major format
- KV cache pre-allocated to max_seq_len to avoid re-allocation
- Zero-init loading avoids allocating 350M random floats

---

## 6. FP16 Inference

### 6.1 DType Propagation

FP16 support required careful dtype propagation through all model components:

- **Embedding**: F32 → F16 conversion after lookup
- **Causal mask**: Created in model's target dtype
- **KV Cache**: Pre-allocated in target dtype
- **RoPE**: Computed in target dtype
- **LayerNorm**: Normalization in F32 for numerical stability, then cast back

### 6.2 Performance

```
┌──────────────────────┬────────┬────────┬────────┐
│ Operation            │ FP32   │ FP16   │ Speedup │
├──────────────────────┼────────┼────────┼────────┤
│ Attention forward    │ 0.8ms  │ 0.6ms  │ 1.3×   │
│ FFN forward          │ 0.5ms  │ 0.4ms  │ 1.2×   │
│ Full block           │ 1.5ms  │ 1.1ms  │ 1.4×   │
│ Prefill (7 tok)      │ 0.50s  │ 0.40s  │ 1.2×   │
│ Per token (gen)      │ 0.15s  │ 0.12s  │ 1.2×   │
└──────────────────────┴────────┴────────┴────────┘
```

### 6.3 When to Use FP16

| Use FP16 | Use FP32 |
|----------|----------|
| Memory constrained | Training (stability) |
| Inference throughput > quality | Numerical precision critical |
| Batch inference | Very small models (<10M params) |

---

## 7. Model Serialization

### 7.1 Safetensors Format

This project uses HuggingFace's [safetensors](https://github.com/huggingface/safetensors)
format for all checkpoint I/O:
- **Zero-copy** tensor loading (mmap-friendly)
- **No pickle** (safe from malicious files)
- **Fast** — no serialization overhead
- **Interoperable** — load in Python/Rust/any safetensors library

### 7.2 Saving

```rust
// Collect tensors with names
let tensors: Vec<(String, TensorView)> = ...;
let data = safetensors::serialize(&tensors, &None)?;
std::fs::write(path, &data)?;
```

### 7.3 Loading

```rust
let data = std::fs::read(path)?;
let tensors = SafeTensors::deserialize(&data)?;
for name in tensor_names {
    let view = tensors.tensor(name)?;
    let shape: Vec<usize> = ...;
    let dtype: DType = ...;
    let tensor = Tensor::from_raw_buffer(view.data(), dtype, &shape, device)?;
}
```

### 7.4 Converting PyTorch Weights

```bash
cargo run --example convert_codegen_to_safetensors -- \
    pytorch_model.bin model.safetensors
```

The converter reads PyTorch pickle format and writes safetensors, optionally
converting dtype (F32 → F16).
