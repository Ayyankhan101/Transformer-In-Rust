use std::path::Path;
use candle_core::{Device, DType, Result, Tensor};
use candle_core::pickle::PthTensors;

use super::config::CodeGenConfig;
use super::model::CodeGenModel;
use crate::layers::attention::MultiHeadAttention;
use crate::layers::embedding::Embedding;
use crate::layers::ffn::{FeedForward, GELUNewFFN};
use crate::layers::norm::LayerNorm;

fn load_tensor(pth: &PthTensors, name: &str, dtype: DType, device: &Device) -> Option<Tensor> {
    let t = pth.get(name).ok()?;
    let t = t?;
    let t = if t.dtype() != dtype { t.to_dtype(dtype).ok()? } else { t };
    t.to_device(device).ok()
}

pub struct WeightLoader;

impl WeightLoader {
    pub fn load_from_pytorch(
        path: &Path,
        config: &CodeGenConfig,
        device: &Device,
    ) -> Result<CodeGenModel> {
        let pth = PthTensors::new(path, None)?;
        let mut model = CodeGenModel::new_blank(config.clone(), device)?;
        let dtype = config.dtype;

        if let Some(w) = load_tensor(&pth, "transformer.wte.weight", dtype, device) {
            model.embedding = Embedding::from_tensor(w);
        }

        for i in 0..config.num_layers {
            let block = &mut model.blocks[i];

            if let Some(w) = load_tensor(&pth, &format!("transformer.h.{i}.ln_1.weight"), dtype, device) {
                let b = load_tensor(&pth, &format!("transformer.h.{i}.ln_1.bias"), dtype, device)
                    .unwrap_or(Tensor::zeros(config.hidden_dim, dtype, device).unwrap());
                block.norm1 = LayerNorm::from_tensor(w, b, config.eps);
            }

            if let Some(w) = load_tensor(&pth, &format!("transformer.h.{i}.attn.qkv_proj.weight"), dtype, device) {
                let w_t = w.transpose(0, 1)?;
                let out_w = load_tensor(&pth, &format!("transformer.h.{i}.attn.out_proj.weight"), dtype, device)
                    .unwrap()
                    .transpose(0, 1)?;
                block.attn = MultiHeadAttention::from_tensors(w_t, out_w, config.num_heads);
            }

            if let Some(w) = load_tensor(&pth, &format!("transformer.h.{i}.ln_2.weight"), dtype, device) {
                let b = load_tensor(&pth, &format!("transformer.h.{i}.ln_2.bias"), dtype, device)
                    .unwrap_or(Tensor::zeros(config.hidden_dim, dtype, device).unwrap());
                block.norm2 = LayerNorm::from_tensor(w, b, config.eps);
            }

            if let Some(fc_in) = load_tensor(&pth, &format!("transformer.h.{i}.mlp.fc_in.weight"), dtype, device) {
                let fc_in_t = fc_in.transpose(0, 1)?;
                let fc_in_b = load_tensor(&pth, &format!("transformer.h.{i}.mlp.fc_in.bias"), dtype, device)
                    .unwrap_or(Tensor::zeros(config.ffn_dim, dtype, device).unwrap());
                let fc_out = load_tensor(&pth, &format!("transformer.h.{i}.mlp.fc_out.weight"), dtype, device)
                    .unwrap();
                let fc_out_t = fc_out.transpose(0, 1)?;
                let fc_out_b = load_tensor(&pth, &format!("transformer.h.{i}.mlp.fc_out.bias"), dtype, device)
                    .unwrap_or(Tensor::zeros(config.hidden_dim, dtype, device).unwrap());
                block.ffn = FeedForward::GELUNew(GELUNewFFN::from_tensors_with_bias(fc_in_t, fc_in_b, fc_out_t, fc_out_b));
            }
        }

        if let Some(w) = load_tensor(&pth, "transformer.ln_f.weight", dtype, device) {
            let b = load_tensor(&pth, "transformer.ln_f.bias", dtype, device)
                .unwrap_or(Tensor::zeros(config.hidden_dim, dtype, device).unwrap());
            model.final_norm = LayerNorm::from_tensor(w, b, config.eps);
        }

        if let Some(w) = load_tensor(&pth, "lm_head.weight", dtype, device) {
            model.lm_head = w.transpose(0, 1)?;
            if let Some(b) = load_tensor(&pth, "lm_head.bias", dtype, device) {
                model.lm_head_bias = b;
            }
        }

        Ok(model)
    }
}
