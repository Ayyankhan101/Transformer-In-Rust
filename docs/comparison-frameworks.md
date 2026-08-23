# Framework Comparison: Rust Transformer vs Alternatives

Comparing this project against other Rust ML frameworks for transformer inference/training.

---

## 1. Overview

| Feature | **This Project** | **candle-examples** | **Burn** | **tch-rs** |
|:--------|:----------------:|:-------------------:|:--------:|:----------:|
| **Runtime** | Pure Rust | Pure Rust | Pure Rust | Libtorch C++ |
| **GPU Support** | ❌ (CPU only) | ✅ (CUDA/Metal) | ✅ (CUDA/Vulkan) | ✅ (CUDA) |
| **Python Deps** | ❌ None | ❌ None | ❌ None | ✅ libtorch |
| **Binary Size** | ~5 MB | ~10 MB | ~8 MB | ~500 MB (libtorch) |
| **Build Time** | ~2 min | ~3 min | ~5 min | ~10 min |
| **Model Support** | GLM + CodeGen | Llama, GPT2, etc. | Custom | Any PyTorch model |
| **Training** | ✅ GLM only | ❌ No | ✅ Yes | ✅ Yes |
| **Quantization** | ❌ FP16 only | ✅ GGML/GGUF | ✅ INT8/FP4 | ❌ No |

---

## 2. Detailed Comparison

### 2.1 This Project (transformer-in-rust)

**Strengths**:
- Pure Rust, zero Python or C++ dependencies
- Minimal binary size (~5 MB release)
- Fast build times (~2 min for release)
- Educational value — everything is hand-written for clarity
- Real CodeGen-350M inference on CPU
- Working training pipeline with safetensors checkpoints
- FP16 support (3.4x speedup on CodeGen-350M, Apple M1 Pro)

**Weaknesses**:
- CPU-only (no CUDA/Metal backend)
- Limited model zoo (GLM + CodeGen only)
- No quantization (FP32 and FP16 only)
- No distributed training

**Best for**: Learning, CPU-only deployment, edge devices, custom architectures

### 2.2 candle-examples (HuggingFace Candle)

**Strengths**:
- Multiple pre-built model examples (Llama, Whisper, Stable Diffusion, etc.)
- CUDA and Metal GPU support
- Quantized inference via GGML/GGUF
- Backed by HuggingFace ecosystem

**Weaknesses**:
- Examples are standalone, not a reusable library
- Training support is limited (no built-in training pipeline)
- Less educational — uses framework APIs, not raw tensor ops

**Best for**: GPU inference, model experimentation, HF model compatibility

### 2.3 Burn

**Strengths**:
- Full training framework with autodiff
- Multiple backends (CUDA, Vulkan, CPU, WebGPU)
- Custom architecture support
- ONNX export/import

**Weaknesses**:
- Newer ecosystem (fewer pre-trained model ports)
- Longer build times due to backend compilation
- More complex API

**Best for**: Training new models, research, multi-backend deployment

### 2.4 tch-rs

**Strengths**:
- Full PyTorch API compatibility
- Any PyTorch model can be loaded and run
- Mature ecosystem, excellent documentation

**Weaknesses**:
- Requires libtorch installation (~500MB-2GB)
- Mixed C++/Rust build process
- Not truly "pure Rust"
- Binary size includes libtorch shared libraries

**Best for**: Production systems already using PyTorch, maximum model compatibility

---

## 3. Performance Benchmarks

*This project measured on Apple M1 Pro, CPU-only, release mode, greedy decoding.
The comparison rows are not re-measured here and are indicative only.*

### CodeGen-350M Inference (FP32)

| Framework | Load Time | Per Token | 64 Tokens |
|:----------|:---------:|:---------:|:---------:|
| **This project** | ~0.6s | 68.8 ms | 4.4s |
| candle (llama example) | 0.8s* | ~120 ms | ~6.5s |
| burn | N/A | N/A | N/A (no CodeGen port) |
| tch-rs | 0.4s | ~100 ms | ~5.2s |

*\* candle-examples uses GGML quantization by default, not FP32*

### FP16 Inference

| Framework | Per Token | Speedup vs FP32 |
|:----------|:---------:|:----------------:|
| **This project** | 20.3 ms | 3.4x |
| tch-rs | ~80 ms | ~1.25x |

### Memory Usage (CodeGen-350M)

| Framework | FP32 Loading | FP32 Runtime | FP16 Runtime |
|:----------|:-----------:|:-----------:|:-----------:|
| **This project** | 760 MB | 1.63 GB | 1.08 GB |
| tch-rs | 700 MB | ~1.2 GB | ~700 MB |

---

## 4. When to Choose What

### Choose this project when:
- **You want to learn**: Hand-written layers are educational
- **CPU-only deployment**: No GPU available or needed
- **Edge/embedded**: Raspberry Pi, IoT, or small devices
- **Binary size matters**: 5 MB release binary
- **No Python deps**: True zero-dependency deployment
- **Quick iteration**: Fast build times

### Choose candle-examples when:
- **GPU available**: CUDA or Metal for fast inference
- **GGML quantization**: 4-bit quantized models for low memory
- **Model variety**: Llama, Mistral, Whisper, etc.
- **HF model hub**: Direct model loading from HuggingFace

### Choose Burn when:
- **Training from scratch**: Full autodiff + training framework
- **Multiple backends**: CPU + GPU + WebGPU in one codebase
- **Custom architectures**: Easy to define new model types
- **Research**: Experiment with novel architectures

### Choose tch-rs when:
- **PyTorch compatibility**: Need to load any existing PyTorch model
- **Production Rust + PyTorch**: Already using libtorch in your stack
- **Maximum performance**: Libtorch's optimized CUDA kernels
- **Mature ecosystem**: Well-documented, widely used

---

## 5. Dependency Comparison

```
        No Python deps         Requires libtorch
        ┌─────────────────┐     ┌──────────────┐
        │ This project    │     │   tch-rs     │
        │ candle-examples │     │              │
        │ Burn            │     │              │
        └─────────────────┘     └──────────────┘

        CPU only               CPU + GPU
        ┌─────────────────┐     ┌──────────────┐
        │ This project    │     │ candle-      │
        │                 │     │ examples     │
        │                 │     │ Burn         │
        │                 │     │ tch-rs       │
        └─────────────────┘     └──────────────┘
```

---

## 6. Summary: One-Line Takeaway

| Framework | One-liner |
|:----------|:----------|
| **This project** | Best for learning + CPU-only edge deployment |
| **candle-examples** | Best for GPU inference with HF ecosystem |
| **Burn** | Best for training + cross-backend deployment |
| **tch-rs** | Best for PyTorch compatibility in production |

---

## 7. Running the Benchmarks

To reproduce these numbers with this project:

```bash
# Full benchmark suite
cargo bench

# Specific benchmarks
cargo bench -- "attention"
cargo bench -- "e2e"
cargo bench -- "f16 vs f32"

# With custom sample size
cargo bench -- --sample-size 50 --measurement-time 10
```
