use candle_core::pickle::PthTensors;
use candle_core::{DType, Device, Error, Result, Tensor};
use std::path::Path;

use super::config::CodeGenConfig;
use super::model::CodeGenModel;
use crate::layers::attention::MultiHeadAttention;
use crate::layers::embedding::Embedding;
use crate::layers::ffn::{FeedForward, GELUNewFFN};
use crate::layers::norm::LayerNorm;

fn to(t: Tensor, dtype: DType, device: &Device) -> Result<Tensor> {
    let t = if t.dtype() != dtype {
        t.to_dtype(dtype)?
    } else {
        t
    };
    t.to_device(device)
}

fn missing(name: &str) -> Error {
    Error::Msg(format!("checkpoint is missing tensor `{name}`"))
}

pub struct WeightLoader;

impl WeightLoader {
    /// Load a HuggingFace PyTorch checkpoint (`pytorch_model.bin`).
    pub fn load_from_pytorch(
        path: &Path,
        config: &CodeGenConfig,
        device: &Device,
    ) -> Result<CodeGenModel> {
        let pth = PthTensors::new(path, None)?;
        let dtype = config.dtype;
        Self::build(
            |name| {
                let t = pth.get(name)?.ok_or_else(|| missing(name))?;
                to(t, dtype, device)
            },
            config,
            device,
        )
    }

    /// Load a safetensors checkpoint (`model.safetensors`), as produced by
    /// `examples/convert_codegen_to_safetensors.rs`.
    pub fn load_from_safetensors(
        path: &Path,
        config: &CodeGenConfig,
        device: &Device,
    ) -> Result<CodeGenModel> {
        let tensors = candle_core::safetensors::load(path, device)?;
        let dtype = config.dtype;
        Self::build(
            |name| {
                let t = tensors.get(name).ok_or_else(|| missing(name))?.clone();
                to(t, dtype, device)
            },
            config,
            device,
        )
    }

    /// Load whichever checkpoint format `path` is, decided by extension.
    pub fn load(path: &Path, config: &CodeGenConfig, device: &Device) -> Result<CodeGenModel> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("safetensors") => Self::load_from_safetensors(path, config, device),
            _ => Self::load_from_pytorch(path, config, device),
        }
    }

    /// Assemble a model from a checkpoint accessor.
    ///
    /// `get` must fail loudly for an absent tensor: a missing one used to be
    /// swallowed, leaving that layer at its all-zero initial value, which
    /// silently zeroes its output instead of reporting a bad checkpoint.
    fn build<F>(get: F, config: &CodeGenConfig, device: &Device) -> Result<CodeGenModel>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        let mut model = CodeGenModel::new_blank(config.clone(), device)?;

        model.embedding = Embedding::from_tensor(get("transformer.wte.weight")?);

        for i in 0..config.num_layers {
            let block = &mut model.blocks[i];

            block.norm1 = LayerNorm::from_tensor(
                get(&format!("transformer.h.{i}.ln_1.weight"))?,
                get(&format!("transformer.h.{i}.ln_1.bias"))?,
                config.eps,
            );

            block.attn = MultiHeadAttention::from_tensors(
                get(&format!("transformer.h.{i}.attn.qkv_proj.weight"))?.transpose(0, 1)?,
                get(&format!("transformer.h.{i}.attn.out_proj.weight"))?.transpose(0, 1)?,
                config.num_heads,
            );

            block.ffn = FeedForward::GELUNew(GELUNewFFN::from_tensors_with_bias(
                get(&format!("transformer.h.{i}.mlp.fc_in.weight"))?.transpose(0, 1)?,
                get(&format!("transformer.h.{i}.mlp.fc_in.bias"))?,
                get(&format!("transformer.h.{i}.mlp.fc_out.weight"))?.transpose(0, 1)?,
                get(&format!("transformer.h.{i}.mlp.fc_out.bias"))?,
            ));
        }

        model.final_norm = LayerNorm::from_tensor(
            get("transformer.ln_f.weight")?,
            get("transformer.ln_f.bias")?,
            config.eps,
        );

        model.lm_head = get("lm_head.weight")?.transpose(0, 1)?;
        model.lm_head_bias = get("lm_head.bias")?;

        Ok(model)
    }
}
