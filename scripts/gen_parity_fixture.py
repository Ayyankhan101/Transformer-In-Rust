#!/usr/bin/env python3
"""Generate tiny CodeGen fixtures for the Rust numerical parity test.

Builds small randomly-initialised CodeGen models with HuggingFace transformers,
saves each state dict in torch pickle format (what `candle_core::pickle::PthTensors`
reads), and records the reference logits for a fixed input sequence.

The fixtures are committed so CI needs no Python. Regenerate with:

    python3 scripts/gen_parity_fixture.py
"""

import json
import pathlib

import torch
from transformers import CodeGenConfig, CodeGenForCausalLM

OUT = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# Two shapes on purpose:
#   n_head == mp_num  -> exactly one head per model-parallel group
#   n_head == 2*mp_num -> several heads per group, so a "one head per group"
#                         shortcut cannot pass the test
VARIANTS = [
    # name,        n_embd, n_layer, n_head, rotary_dim
    ("tiny_h4", 128, 2, 4, 16),   # head_dim 32, partial rotary
    ("tiny_h8", 128, 2, 8, 16),   # head_dim 16, full rotary
]

TOKENS = [3, 17, 42, 8, 255, 1, 99, 128]


def build(name, n_embd, n_layer, n_head, rotary_dim):
    torch.manual_seed(0)
    config = CodeGenConfig(
        vocab_size=256,
        n_positions=64,
        n_ctx=64,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        rotary_dim=rotary_dim,
        n_inner=None,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        tie_word_embeddings=False,
    )
    model = CodeGenForCausalLM(config)
    model.eval()

    # LayerNorm weights default to exactly 1.0 / bias 0.0, which a buggy loader
    # that silently drops tensors could accidentally match. Perturb them so a
    # missed norm tensor shows up as a logit mismatch.
    torch.manual_seed(1)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.add_(torch.randn_like(module.weight) * 0.05)
                module.bias.add_(torch.randn_like(module.bias) * 0.05)

    input_ids = torch.tensor([TOKENS], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids).logits

    torch.save(model.state_dict(), OUT / f"{name}.pth")
    (OUT / f"{name}_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True, default=str) + "\n"
    )
    (OUT / f"{name}_logits.json").write_text(
        json.dumps(
            {
                "tokens": TOKENS,
                "shape": list(logits.shape),
                "logits": logits.reshape(-1).tolist(),
            }
        )
        + "\n"
    )
    print(f"{name}: logits {tuple(logits.shape)}  head_dim {n_embd // n_head}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        build(*variant)
