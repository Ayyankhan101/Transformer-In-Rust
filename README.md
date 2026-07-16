<div align="center">

# 🧠 Transformer in Rust

### From-scratch transformer architectures powered by [Candle](https://github.com/huggingface/candle)

**GLM** (blank-infilling) · **CodeGen-350M** (causal code generation) · CPU-only inference

[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![Candle](https://img.shields.io/badge/Candle-0.8-blue)](https://github.com/huggingface/candle)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-20%2F20%20✓-brightgreen)]()

<br>

*No GPU required. No Python runtime. Pure Rust tensor ops — 350M params on an i5-6600.*

</div>

---

## 📋 Overview

Two transformer models share a hand-written layer library built with raw tensor operations:

| Model | Architecture | Params | Use Case |
|:------|:-------------|:------:|:---------|
| **GLM** | Blank-infilling | ~14M | Training playground / demo |
| **CodeGen-350M** | Causal decoder-only | 350M | Real code generation with pretrained weights |

<br>

```mermaid
graph LR
    subgraph Shared["🔧 Shared Layer Library"]
        E[Embedding]
        MHA[Multi-Head Attention]
        FFN[Feed-Forward Network<br/>SwiGLU / GELU]
        LN[LayerNorm / RMSNorm]
        TB[Transformer Block]
    end

    subgraph GLM["🧠 GLM Model"]
        G1[2D Positional Encoding]
        G2[Blank-Infilling Mask]
        G3[Autoregressive Training]
    end

    subgraph CodeGen["⚡ CodeGen-350M"]
        C1[RoPE Rotary Embedding]
        C2[KV Cache]
        C3[Parallel Attn + FFN]
    end

    E --> TB
    MHA --> TB
    FFN --> TB
    LN --> TB

    TB --> GLM
    TB --> CodeGen

    style Shared fill:#1a1a2e,stroke:#e94560,color:#fff
    style GLM fill:#16213e,stroke:#0f3460,color:#fff
    style CodeGen fill:#0f3460,stroke:#e94560,color:#fff
```

---

## 🏗️ Architecture

### Transformer Block (Pre-Norm)

```mermaid
graph TD
    X["x"] --> Norm1["RMSNorm"]
    Norm1 --> Attn["Multi-Head Attention"]
    X --> Add1["+ (Residual)"]
    Attn --> Add1
    Add1 --> Norm2["RMSNorm"]
    Norm2 --> FFN["Feed-Forward (SwiGLU/GELU)"]
    Add1 --> Add2["+ (Residual)"]
    FFN --> Add2
    Add2 --> Out["Output"]

    style Norm1 fill:#e94560,color:#fff
    style Norm2 fill:#e94560,color:#fff
    style Attn fill:#0f3460,color:#fff
    style FFN fill:#16213e,color:#fff
    style Add1 fill:#533483,color:#fff
    style Add2 fill:#533483,color:#fff
```

### CodeGen: Parallel Sub-Block Design

```mermaid
graph TD
    X["x"] --> LN["LayerNorm"]
    LN --> Attn["Attention(x)"]
    LN --> FFN["FFN(x)"]
    X --> Res["+"]
    Attn --> Res
    FFN --> Res
    Res --> Out["Output"]

    style LN fill:#e94560,color:#fff
    style Attn fill:#0f3460,color:#fff
    style FFN fill:#16213e,color:#fff
    style Res fill:#533483,color:#fff
```

> Both attention and FFN operate on the **same normalized input** — outputs sum with residual in one step (GPT-J style).

### GLM: Blank-Infilling Flow

```mermaid
graph LR
    Input["Text + [MASK] tokens"] --> Enc["2D Positional Encoding<br/>pos_1 + pos_2"]
    Enc --> Mask["Blank-Infilling<br/>Attention Mask"]
    Mask --> Fwd["Forward Pass"]
    Fwd --> Loss["Cross-Entropy Loss<br/>(corrupted positions only)"]

    style Input fill:#0f3460,color:#fff
    style Enc fill:#533483,color:#fff
    style Mask fill:#e94560,color:#fff
    style Loss fill:#16213e,color:#fff
```

---

## 🚀 CodeGen Inference Pipeline

```mermaid
graph TD
    Prompt["User Prompt"] --> Tokenize["BPE Tokenize"]
    Tokenize --> Prefill["Prefill: KV Cache Population"]
    Prefill --> Sample["Sample: Last Token Logit"]
    Sample --> Loop{"EOS or max tokens?"}
    Loop -->|No| Forward["Forward Pass (single token)"]
    Forward --> KV["Append to KV Cache"]
    KV --> RepPen["Repetition Penalty"]
    RepPen --> Temp["Temperature Scaling"]
    Temp --> TopK["Top-K Filtering (40)"]
    TopK --> TopP["Top-P Nucleus (0.9)"]
    TopP --> Sample
    Loop -->|Yes| Done["Output Generated Code"]

    style Prompt fill:#0f3460,color:#fff
    style Prefill fill:#533483,color:#fff
    style Loop fill:#e94560,color:#fff
    style Done fill:#16213e,color:#fff
```

---

## ⚡ Performance

*Benchmarked on i5-6600 (4C/4T, 7.5GB RAM) — release mode*

| Operation | Time |
|:----------|:-----|
| Weight loading (350M) | ~0.5s |
| Prefill (7 tokens) | ~0.3s |
| Autoregressive step | ~0.1s/token |
| 37-token generation | ~3.2s |
| GLM training step | ~0.02s |

> ⚠️ Debug builds are ~20× slower.

---

## 📁 Project Structure

```mermaid
graph TD
    Main["main.rs<br/>CLI Dispatcher"]
    Tok["tokenizer.rs<br/>BPE Wrapper"]

    subgraph Layers["src/layers/"]
        Att["attention.rs<br/>Fused QKV"]
        Blk["block.rs<br/>Transformer Block"]
        Emb["embedding.rs<br/>Token Lookup"]
        FFN["ffn.rs<br/>SwiGLU + GELU"]
        Norm["norm.rs<br/>LayerNorm + RMSNorm"]
    end

    subgraph GLMMod["src/glm/"]
        GC["config.rs"]
        GM["model.rs"]
        GAM["attention_mask.rs"]
        GP["positions.rs"]
    end

    subgraph CGMod["src/codegen/"]
        CC["config.rs"]
        CM["model.rs"]
        CR["rotary.rs<br/>RoPE"]
        CW["weights.rs<br/>PyTorch Loader"]
    end

    subgraph GenMod["src/generation/"]
        CG["codegen_generate.rs<br/>Autoregressive + Sampling"]
        GGen["glm_generate.rs"]
    end

    subgraph TrainMod["src/training/"]
        TR["train.rs<br/>GLM Training Loop"]
    end

    Main --> Tok
    Main --> Layers
    Main --> GLMMod
    Main --> CGMod
    Main --> GenMod
    Main --> TrainMod

    style Layers fill:#1a1a2e,stroke:#e94560,color:#fff
    style GLMMod fill:#16213e,stroke:#0f3460,color:#fff
    style CGMod fill:#0f3460,stroke:#e94560,color:#fff
    style GenMod fill:#533483,stroke:#e94560,color:#fff
    style TrainMod fill:#1a1a2e,stroke:#0f3460,color:#fff
```

---

## 🛠️ Quick Start

### Prerequisites

- Rust 2021 edition
- CodeGen-350M weights (797MB)

### Download Weights

```bash
hf download Salesforce/codegen-350M-multi --local-dir codegen_weights
```

### Build & Run

```bash
# Build in release mode (important — debug is 20x slower)
cargo build --release

# Conversational code generation (multi-turn)
cargo run --release -- chat

# Single-shot code generation
cargo run --release -- complete "def fibonacci(n):"

# Interactive REPL
cargo run --release -- repl

# Model info and weight status
cargo run --release -- info

# Download CodeGen-350M weights
cargo run --release -- download

# HTTP inference server
cargo run --release --features server -- serve --port 8080

# GLM training demo
cargo run --release -- glm-train --data-path data --steps 500

# Global flags
#   --f16              Use FP16 precision
#   --weights-dir DIR  Path to weights (default: codegen_weights)
```

---

## 🧩 Model Configurations

### GLM (Tiny)

| Parameter | Value |
|:----------|:------|
| hidden_dim | 256 |
| num_layers | 6 |
| num_heads | 8 |
| ffn_dim | 1024 |
| vocab_size | 16,384 |
| ~params | ~14M |

### CodeGen-350M

| Parameter | Value |
|:----------|:------|
| hidden_dim | 1024 |
| num_layers | 20 |
| num_heads | 16 |
| ffn_dim | 4096 |
| rotary_dim | 32 |
| vocab_size | 51,200 |
| ~params | ~350M |

---

## 🔧 Key Features

- **[RoPE](https://arxiv.org/abs/2104.09864)** — Rotary Position Embedding with even-odd pair rotation
- **[KV Cache](https://arxiv.org/abs/1901.02860)** — Autoregressive generation without recomputation
- **SwiGLU** — Gated feed-forward for GLM
- **Parallel Sub-Blocks** — GPT-J style attn + FFN in parallel
- **Blank-Infilling** — Bidirectional context with causal within-blank masking
- **Sampling Pipeline** — Repetition penalty → temperature → top-k → top-p → random sample
- **Zero-Init Loading** — Avoids allocating 350M random floats before overwriting with weights

---

## 📦 Dependencies

| Crate | Version | Purpose |
|:------|:-------:|:--------|
| [candle-core](https://crates.io/crates/candle-core) | 0.8 | Tensor ops, CPU backend, pickle loader |
| [candle-nn](https://crates.io/crates/candle-nn) | 0.8 | Softmax, cross-entropy |
| [tokenizers](https://crates.io/crates/tokenizers) | 0.21 | HuggingFace BPE tokenizer |
| [anyhow](https://crates.io/crates/anyhow) | 1.0 | Error handling |
| [serde_json](https://crates.io/crates/serde_json) | 1.0 | Config parsing |

---

## 🧪 Testing

```bash
cargo test
```

**20 tests** covering:

| Module | Tests |
|:-------|:------|
| Embedding | Lookup correctness |
| Attention | Causal mask shape + triangularity |
| FFN | GELU forward, SwiGLU gate |
| Norm | LayerNorm, RMSNorm forward |
| GLM | Causal forward, blank-infill forward, training loss, attention mask, safetensors save/load |
| CodeGen | Blank-forward, RoPE no-segfault |
| Quantized | INT8 quantized linear roundtrip, ranking preservation |
| Sampling | Argmax, temperature-zero |
| Training | LR scheduler (cosine/linear/constant) |

---

## 📜 License

MIT

---

<div align="center">

**Built with ❤️ in Rust**

*No Python. No GPU. Just tensors.*

</div>
