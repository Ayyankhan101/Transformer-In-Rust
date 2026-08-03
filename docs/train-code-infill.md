# Tutorial: Train a Code Infill Model on Your Laptop

**Goal**: Train a GLM (blank-infilling) transformer from scratch on your laptop CPU — no GPU required.

**Time**: ~30 minutes
**Hardware**: Any modern x86_64 laptop (tested on i5-6600, 7.5GB RAM)

---

## 1. What You'll Build

GLM (General Language Model) is a blank-infilling transformer: it learns to predict masked spans of text using bidirectional context. Unlike causal models that only see left-to-right, GLM corrupts spans in the input and trains the model to reconstruct them.

By the end of this tutorial, you'll have:
- A trained GLM model that can fill in code blanks
- Checkpoints saved in safetensors format
- A generation demo showing the model in action

---

## 2. Prerequisites

```bash
# Rust toolchain
rustc --version   # Should be 1.70+
cargo --version

# Clone this repo
git clone <your-repo-url>
cd transformer-in-rust
```

---

## 3. Training Configuration

The training pipeline uses YAML configuration. Here's the default:

```yaml
# configs/train.yaml
model:
  hidden_dim: 256
  num_layers: 6
  num_heads: 8
  ffn_dim: 1024
  max_seq_len: 128
  vocab_size: 16384

training:
  batch_size: 8
  learning_rate: 0.001
  max_steps: 5000
  warmup_steps: 200
  lr_schedule: cosine
  eval_interval: 100
  save_interval: 500
  keep_last_n: 3
  max_seq_len: 128
  gradient_accumulation_steps: 4
  max_grad_norm: 1.0
  data_dir: training_data
  download_if_empty: true
```

### Key Parameters

| Parameter | Value | What it controls |
|-----------|-------|------------------|
| `hidden_dim` | 256 | Width of the transformer (small = fast) |
| `num_layers` | 6 | Depth of the transformer |
| `max_steps` | 5000 | Total training steps (~14M param model) |
| `gradient_accumulation_steps` | 4 | Simulates larger batch size on limited memory |
| `warmup_steps` | 200 | Gradually increases LR to avoid instability |

---

## 4. Running Training

### Start training with YAML config:

```bash
cargo run --release -- glm-train --data-path training_data --steps 5000
```

### Or with defaults (auto-downloads training data):

```bash
cargo run --release -- glm-train
```

### What you'll see:

```
Step    100 | loss=4.21 | lr=0.00050 | 0.12s/step
Step    200 | loss=3.85 | lr=0.00100 | 0.11s/step
Step    300 | loss=3.52 | lr=0.00087 | 0.11s/step
...
Step   1000 | loss=2.14 | lr=0.00012 | 0.11s/step
Step   2000 | loss=1.43 | lr=0.00004 | 0.11s/step
...
Eval  | train_loss=1.02 | val_loss=1.31
```

Loss should drop steadily from ~4.5 to ~1.0 over 5000 steps.

### Checkpoints

Checkpoints are saved to `checkpoints/`:
```
checkpoints/
├── model_step_0000500.safetensors
├── optimizer_step_0000500.json
├── model_step_0001000.safetensors
├── ...
└── training_state.json
```

---

## 5. Training Data

By default, the trainer downloads CPython's `functools.py` (~500 lines of Python). You can also provide your own data:

```
training_data/
├── train/       # Text files for training
└── eval/        # Text files for evaluation
```

The DataLoader splits data at the file level (80/20 train/eval by default), then extracts random subsequences of length `max_seq_len`.

---

## 6. Testing the Trained Model

After training completes, test with:

```bash
cargo run --release -- complete "def fibonacci(n):"
```

This loads the latest checkpoint and generates code completions.

---

## 7. Performance Notes

| Hardware | Step Time | 5000 Steps | Memory |
|----------|-----------|------------|--------|
| i5-6600 (4C/4T) | ~0.11s | ~9 min | ~500 MB |
| i7-12700 (12C/16T) | ~0.04s | ~3 min | ~500 MB |
| M1 MacBook Air | ~0.03s | ~2.5 min | ~400 MB |

> Training uses **no GPU**. Pure CPU tensor operations via Candle.

---

## 8. Next Steps

1. **Experiment with hyperparameters**: Try `hidden_dim: 512` or `num_layers: 12` for a larger model
2. **Use your own data**: Point `data_dir` at any directory of text/code files
3. **Extend vocabulary**: Train a custom BPE tokenizer for your domain
4. **Convert to safetensors**: Use the checkpoint's built-in safetensors format for inference

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `No data files found` | Empty data directory | Use `--data-path` with a valid directory or populate `training_data/` |
| Loss not decreasing | Learning rate too high/low | Adjust `learning_rate` (try 3e-4) |
| Out of memory | Batch/sequence too large | Reduce `batch_size` or `max_seq_len` |
| NaN loss | Gradient explosion | Reduce `learning_rate` or increase `warmup_steps` |
| Slow training | Debug build | Always use `--release` for training |
