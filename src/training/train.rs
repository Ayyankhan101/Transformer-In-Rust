use std::io::Read;
use std::path::Path;

use candle_core::{Device, Result, Tensor};
use candle_nn::{AdamW, ParamsAdamW, Optimizer};

use crate::glm::config::GLMConfig;
use crate::glm::trainable::TrainableGLMModel;
use crate::tokenizer::CodeGenTokenizer;

const DEFAULT_DATA_URL: &str = "https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/functools.py";

struct TrainingExample {
    tokens: Vec<u32>,
}

fn load_data(data_dir: &Path, tokenizer: &CodeGenTokenizer) -> Result<Vec<TrainingExample>> {
    let mut examples = Vec::new();

    let entries = match std::fs::read_dir(data_dir) {
        Ok(e) => e,
        Err(_) => return Ok(examples),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("py") {
            let content = match std::fs::read_to_string(&path) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let ids = match tokenizer.encode(&content) {
                Ok(ids) => ids,
                Err(_) => continue,
            };
            if ids.len() > 10 {
                examples.push(TrainingExample { tokens: ids });
            }
        }
    }

    Ok(examples)
}

fn download_default_data(dest_dir: &Path) -> Result<()> {
    println!("  Downloading sample Python code from CPython repository...");
    let response = ureq::get(DEFAULT_DATA_URL)
        .call()
        .map_err(|e| candle_core::Error::Msg(format!("Download failed: {e}")))?;

    let mut body = Vec::new();
    let mut body_owned = response.into_body();
    let mut reader = body_owned
        .as_reader();
    reader
        .read_to_end(&mut body)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read response: {e}")))?;

    let dest_path = dest_dir.join("functools.py");
    std::fs::write(&dest_path, &body)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write {:?}: {e}", dest_path)))?;
    println!("  Saved to {:?} ({} bytes)", dest_path, body.len());
    Ok(())
}

fn cross_entropy_loss(logits: &Tensor, labels: &[i64]) -> Result<Tensor> {
    let seq_len = logits.dims()[1];
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for i in 0..seq_len {
        if labels[i] >= 0 {
            let logits_i = logits.get(0)?.get(i)?;
            let ce = candle_nn::ops::log_softmax(&logits_i, 0)?
                .get(labels[i] as usize)?
                .neg()?;
            total_loss += ce.to_scalar::<f32>()?;
            count += 1;
        }
    }

    if count > 0 {
        total_loss /= count as f32;
    }

    Ok(Tensor::new(total_loss, logits.device())?)
}

pub struct GLMTrainer {
    model: TrainableGLMModel,
    config: GLMConfig,
    optimizer: AdamW,
    step: usize,
    tokenizer: CodeGenTokenizer,
}

impl GLMTrainer {
    pub fn new(config: GLMConfig, device: &Device) -> Result<Self> {
        let model = TrainableGLMModel::new(config.clone(), device)?;
        let params = model.param_vars();
        let optimizer = AdamW::new(
            params,
            ParamsAdamW {
                lr: 1e-4,
                weight_decay: 0.01,
                ..Default::default()
            },
        )?;
        let tokenizer = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json")
            .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self { model, config, optimizer, step: 0, tokenizer })
    }

    pub fn train_step(&mut self, token_ids: &[u32], _device: &Device) -> Result<f64> {
        let n = token_ids.len();
        let mask_token_id = self.config.vocab_size as u32 - 1;

        let mut rng_state = self.step as u64;
        let mut rand = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state as f64) / (u64::MAX as f64)
        };

        let mut input_ids = token_ids.to_vec();
        let mut labels = vec![-1i64; n];

        for i in 0..n {
            let r = rand();
            if r < self.config.blank_ratio {
                labels[i] = token_ids[i] as i64;
                if rand() < self.config.mask_ratio {
                    input_ids[i] = mask_token_id;
                } else {
                    input_ids[i] = (rand() * self.config.vocab_size as f64) as u32;
                }
            }
        }

        let logits = self.model.forward_causal(&input_ids)?;
        let loss = cross_entropy_loss(&logits, &labels)?;

        let loss_scalar = loss.to_scalar::<f32>()?;
        self.optimizer.backward_step(&loss)?;

        self.step += 1;
        Ok(loss_scalar as f64)
    }

    pub fn train(
        &mut self,
        data_dir: &Path,
        num_steps: usize,
        download_if_empty: bool,
        device: &Device,
    ) -> Result<()> {
        let mut examples = load_data(data_dir, &self.tokenizer)?;

        if examples.is_empty() {
            if download_if_empty {
                println!("No data found in {:?}, downloading sample data...", data_dir);
                download_default_data(data_dir)?;
                examples = load_data(data_dir, &self.tokenizer)?;
            }
            if examples.is_empty() {
                println!("No training data available. Using dummy data for testing.");
            }
        } else {
            println!("Loaded {} training file(s)", examples.len());
        }

        for s in 0..num_steps {
            let tokens = if examples.is_empty() {
                let dummy: Vec<u32> = (100..120).collect();
                dummy
            } else {
                let ex = &examples[s % examples.len()];
                let max_len = self.config.max_seq_len;
                if ex.tokens.len() > max_len {
                    let start = (s * 17) % (ex.tokens.len() - max_len);
                    ex.tokens[start..start + max_len].to_vec()
                } else {
                    ex.tokens.clone()
                }
            };

            let loss = self.train_step(&tokens, device)?;
            if s % 10 == 0 {
                println!("Step {}: loss = {:.4}", self.step(), loss);
            }

            if s > 0 && s % 100 == 0 {
                self.save_checkpoint()?;
            }
        }

        self.save_checkpoint()?;
        println!("Training complete.");

        Ok(())
    }

    pub fn save_checkpoint(&self) -> Result<()> {
        let dir = Path::new("glm_checkpoint");
        std::fs::create_dir_all(dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create checkpoint dir: {e}")))?;

        let params = self.model.param_vars();
        let names = param_names();
        for (i, var) in params.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
            let fname = dir.join(format!("{:03}_{}.npy", i, name.replace('.', "_")));
            var.as_tensor().write_npy(&fname)?;
        }

        let step_path = dir.join("step.txt");
        std::fs::write(&step_path, format!("{}", self.step))
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write step: {e}")))?;

        println!("  Checkpoint saved to {:?} (step {})", dir, self.step);
        Ok(())
    }

    pub fn load_checkpoint(&mut self) -> Result<()> {
        let dir = Path::new("glm_checkpoint");
        if !dir.exists() {
            return Ok(());
        }

        let step_path = dir.join("step.txt");
        if step_path.exists() {
            let content = std::fs::read_to_string(&step_path)
                .map_err(|e| candle_core::Error::Msg(format!("Failed to read step: {e}")))?;
            self.step = content.trim().parse().unwrap_or(0);
        }

        let params = self.model.param_vars();
        let names = param_names();
        for (i, var) in params.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
            let fname = dir.join(format!("{:03}_{}.npy", i, name.replace('.', "_")));
            if fname.exists() {
                let tensor = Tensor::read_npy(&fname)?;
                var.set(&tensor)?;
            }
        }

        println!("Loaded checkpoint from {:?} (step {})", dir, self.step);
        Ok(())
    }

    pub fn step(&self) -> usize {
        self.step
    }
}

fn param_names() -> Vec<String> {
    let mut names = Vec::new();
    names.push("embedding.weight".to_string());
    names.push("pos_1_embedding".to_string());
    names.push("pos_2_embedding".to_string());
    for i in 0..6 {
        names.push(format!("h.{i}.norm1.weight"));
        names.push(format!("h.{i}.attn.qkv_weight"));
        names.push(format!("h.{i}.attn.out_weight"));
        names.push(format!("h.{i}.norm2.weight"));
        names.push(format!("h.{i}.mlp.up"));
        names.push(format!("h.{i}.mlp.gate"));
        names.push(format!("h.{i}.mlp.down"));
    }
    names.push("final_norm.weight".to_string());
    names.push("lm_head".to_string());
    names
}
