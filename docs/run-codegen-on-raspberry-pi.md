# Tutorial: Run CodeGen-350M on a Raspberry Pi

**Goal**: Run a 350M-parameter code generation model on a $35 Raspberry Pi — no GPU, no cloud, no Python.

**Hardware**: Raspberry Pi 4/5 (4GB+ RAM recommended)
**Time**: ~30 minutes setup

---

## 1. Why This Matters

Running a 350M transformer on a Pi demonstrates:
- Pure-CPU inference is viable for modern LLMs
- Rust + Candle provides excellent performance without Python overhead
- Edge-based code generation is possible (privacy, offline, low cost)

---

## 2. Prerequisites

```bash
# Raspberry Pi OS (64-bit recommended)
uname -m   # Should show aarch64

# Install Rust on Pi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

---

## 3. Download Weights

On your Pi (or a desktop, then copy via USB/SSH):

```bash
# Install huggingface-cli or use direct download
pip install huggingface-hub
huggingface-cli download Salesforce/codegen-350M-multi \
    --local-dir codegen_weights
```

**Expected size**: ~797MB for `pytorch_model.bin`

---

## 4. Convert to Safetensors (Optional)

For faster loading, convert to safetensors:

```bash
# On your desktop (faster), then copy the .safetensors file to Pi
cargo run --example convert_codegen_to_safetensors -- \
    codegen_weights/pytorch_model.bin \
    codegen_weights/model.safetensors
```

Safetensors loading is ~2× faster than PyTorch pickle loading.

---

## 5. Build & Run

```bash
# Clone and build (this takes ~10 minutes on Pi 4)
git clone <your-repo-url>
cd transformer-in-rust
cargo build --release

# Run with CodeGen
cargo run --release -- complete "def fibonacci(n):"

# Or use the interactive REPL
cargo run --release -- repl
```

---

## 6. Expected Performance

### Raspberry Pi 4 (4GB)

| Operation | Time |
|-----------|------|
| Build (first time) | ~15 min |
| Weight loading | ~8s (pickle) / ~4s (safetensors) |
| Prefill (7 tokens) | ~2.5s |
| Per token | ~0.8s |
| 50-token generation | ~45s |

### Raspberry Pi 5 (8GB)

| Operation | Time |
|-----------|------|
| Build | ~5 min |
| Weight loading | ~3s |
| Prefill (7 tokens) | ~0.8s |
| Per token | ~0.3s |
| 50-token generation | ~16s |

---

## 7. Memory Optimization

### Use FP16 (half precision)

```bash
cargo run --release -- --f16 complete "def fibonacci(n):"
```

This reduces memory by ~40% with minimal quality loss:
- Model: 700MB (FP32) → 350MB (FP16)
- Runtime: ~1GB → ~600MB

### Reduce max sequence length

If memory constrained, edit `src/codegen/config.rs`:
```rust
max_seq_len: 256,   // Default is 2048
```

---

## 8. Running as a Service

Create a systemd service for automatic startup:

```bash
sudo cat > /etc/systemd/system/codegen.service << 'EOF'
[Unit]
Description=CodeGen Inference Server
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/transformer-in-rust
ExecStart=/home/pi/transformer-in-rust/target/release/rust_transformer repl
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable codegen
sudo systemctl start codegen
```

---

## 9. Tips

1. **Use `--release` always**: Debug builds are 20× slower
2. **ARM-specific tuning**: Candle uses ARM NEON intrinsics automatically
3. **SSD over SD card**: Weight loading is I/O bound; an external SSD helps
4. **Swap**: If using a 4GB Pi, 1GB swap is helpful for the build step
5. **Cross-compile**: Build on a fast desktop, copy the binary to Pi:
   ```bash
   # On desktop
   rustup target add aarch64-unknown-linux-gnu
   cargo build --release --target aarch64-unknown-linux-gnu
   scp target/aarch64-unknown-linux-gnu/release/rust_transformer pi@raspberrypi:~
   ```

---

## 10. Limitations

- Generation speed is ~0.3-0.8 tokens/second (usable for short completions)
- Not suitable for real-time interactive use
- Large context windows (>512 tokens) may cause OOM
- `--f16` halves weight memory and is wired through every layer
