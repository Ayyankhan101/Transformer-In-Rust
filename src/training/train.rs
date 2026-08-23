use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use bytemuck;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Result, Tensor, Var};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use half;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::glm::config::GLMConfig;
use crate::glm::trainable::TrainableGLMModel;
use crate::tokenizer::CodeGenTokenizer;
use crate::training::config::{TrainConfig, TrainingConfig};
use crate::training::data::{
    download_default_data, split_train_eval, DataLoader, TrainingExample as DataTrainingExample,
};
use crate::training::loss::cross_entropy_loss;
use crate::training::lr_scheduler::LrScheduler;

#[allow(dead_code)]
const EOS_TOKEN_ID: u32 = 50256;

pub struct GLMTrainer {
    model: TrainableGLMModel,
    config: TrainingConfig,
    glm_config: GLMConfig,
    optimizer: AdamW,
    lr_scheduler: LrScheduler,
    step: usize,
    tokenizer: CodeGenTokenizer,
    dtype: DType,
    loss_history: VecDeque<f64>,
    rng: StdRng,
}

/// Corrupt a sequence for masked denoising: some positions are replaced by the
/// mask token or a random token, and their originals become the labels.
/// Positions left alone get `-1`, which [`cross_entropy_loss`] ignores.
fn corrupt(config: &GLMConfig, tokens: &[u32], rng: &mut impl Rng) -> (Vec<u32>, Vec<i64>) {
    let mask_token_id = config.vocab_size as u32 - 1;
    let mut inputs = tokens.to_vec();
    let mut labels = vec![-1i64; tokens.len()];

    for i in 0..tokens.len() {
        if rng.gen::<f64>() < config.blank_ratio {
            labels[i] = tokens[i] as i64;
            inputs[i] = if rng.gen::<f64>() < config.mask_ratio {
                mask_token_id
            } else {
                rng.gen_range(0..config.vocab_size as u32)
            };
        }
    }
    (inputs, labels)
}

impl GLMTrainer {
    pub fn from_config(train_config: &TrainConfig, device: &Device) -> Result<Self> {
        let glm_config = train_config.to_glm_config();
        let tc = &train_config.training;
        let dtype = match tc.dtype.as_str() {
            "f16" => DType::F16,
            _ => DType::F32,
        };

        let model = TrainableGLMModel::new(glm_config.clone(), device)?;
        let params = model.param_vars();

        let optimizer = AdamW::new(
            params,
            ParamsAdamW {
                lr: tc.learning_rate,
                weight_decay: tc.weight_decay,
                beta1: tc.beta1,
                beta2: tc.beta2,
                eps: tc.eps,
            },
        )?;

        let lr_scheduler = LrScheduler::new(tc.learning_rate, &tc.lr_schedule);

        let tokenizer = CodeGenTokenizer::from_file(
            tc.tokenizer_path
                .to_str()
                .unwrap_or("codegen_weights/tokenizer.json"),
        )
        .map_err(|e| candle_core::Error::Msg(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            model,
            config: tc.clone(),
            glm_config,
            optimizer,
            lr_scheduler,
            step: 0,
            tokenizer,
            dtype,
            loss_history: VecDeque::with_capacity(100),
            rng: StdRng::seed_from_u64(tc.seed),
        })
    }

    /// Corrupt, forward, loss. The returned tensor stays connected to the
    /// autograd graph — the caller owns the optimizer step.
    pub fn compute_loss(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        let (input_ids, labels) = corrupt(&self.glm_config, token_ids, &mut self.rng);
        let logits = self.model.forward_causal(&input_ids)?;
        cross_entropy_loss(&logits, &labels)
    }

    /// Backward, clip by global norm, step. Advances the step counter and the
    /// learning-rate schedule.
    fn apply_gradients(&mut self, loss: &Tensor) -> Result<()> {
        let mut grads = loss.backward()?;
        let params = self.model.param_vars();

        if self.config.max_grad_norm > 0.0 {
            clip_grad_norm(&mut grads, &params, self.config.max_grad_norm)?;
        }

        // Set the rate this step will use, then step — the previous code set it
        // afterwards, so every step ran on the preceding step's rate.
        self.optimizer.set_learning_rate(self.lr_scheduler.get_lr());
        self.optimizer.step(&grads)?;

        self.step += 1;
        self.lr_scheduler.step();
        Ok(())
    }

    /// One optimizer step over `gradient_accumulation_steps` micro-batches of
    /// `micro_batch_size` sequences each.
    ///
    /// Averaging the micro-batch losses and running a single backward is
    /// equivalent to accumulating their gradients, and needs no `GradStore`
    /// bookkeeping.
    pub fn train_step(&mut self, loader: &mut DataLoader, _device: &Device) -> Result<f64> {
        let accum_steps = self.config.gradient_accumulation_steps.max(1);
        let mut losses: Vec<Tensor> = Vec::new();

        for _ in 0..accum_steps {
            for tokens in loader.next_batch(self.config.micro_batch_size) {
                losses.push(self.compute_loss(&tokens)?);
            }
        }

        if losses.is_empty() {
            return Ok(0.0);
        }

        let mut total = losses[0].clone();
        for loss in &losses[1..] {
            total = (total + loss)?;
        }
        let loss = (total / losses.len() as f64)?;
        let loss_scalar = loss.to_scalar::<f32>()? as f64;

        self.apply_gradients(&loss)?;

        self.loss_history.push_back(loss_scalar);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }

        Ok(loss_scalar)
    }

    pub fn train(&mut self, data_dir: &Path, device: &Device) -> Result<()> {
        // Load data
        let mut examples = load_data(data_dir, &self.tokenizer)?;

        if examples.is_empty() {
            if self.config.download_if_empty {
                println!(
                    "No data found in {:?}, downloading sample data...",
                    data_dir
                );
                download_default_data(data_dir)?;
                examples = load_data(data_dir, &self.tokenizer)?;
            }
            if examples.is_empty() {
                println!("No training data available. Using dummy data for testing.");
            }
        } else {
            println!("Loaded {} training file(s)", examples.len());
        }

        // Split train/eval
        let (train_examples, eval_examples) = split_train_eval(&examples, self.config.train_split);
        println!(
            "Train examples: {}, Eval examples: {}",
            train_examples.len(),
            eval_examples.len()
        );

        // Create data loaders
        let mut train_loader =
            DataLoader::from_examples(train_examples, self.config.max_seq_len, self.config.seed)?;

        let mut eval_loader = DataLoader::from_examples(
            eval_examples,
            self.config.max_seq_len,
            self.config.seed + 1,
        )?;

        // Try to load checkpoint
        self.load_checkpoint()?;

        println!("Starting training from step {}", self.step);
        println!("  Max steps: {}", self.config.max_steps);
        println!("  Learning rate: {}", self.config.learning_rate);
        println!("  Eval every: {} steps", self.config.eval_every);
        println!("  Save every: {} steps", self.config.save_every);
        println!("  Dtype: {:?}", self.dtype);

        // Training loop
        while self.step < self.config.max_steps {
            {
                let loss = self.train_step(&mut train_loader, device)?;

                // Logging
                if self.step % self.config.log_every == 0 {
                    let avg_loss: f64 =
                        self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64;
                    let current_lr = self.lr_scheduler.get_lr();
                    println!(
                        "Step {}: loss = {:.4} (avg = {:.4}), lr = {:.2e}",
                        self.step, loss, avg_loss, current_lr
                    );
                }

                // Evaluation
                if self.step % self.config.eval_every == 0 && self.step > 0 {
                    let eval_loss = self.evaluate(&mut eval_loader, device)?;
                    println!("  Eval loss: {:.4}", eval_loss);
                    eval_loader.reset();
                }

                // Checkpoint
                if self.step % self.config.save_every == 0 {
                    self.save_checkpoint()?;
                    self.cleanup_old_checkpoints()?;
                }

                if self.step >= self.config.max_steps {
                    break;
                }
            }
        }

        // Final save
        self.save_checkpoint()?;
        println!("Training complete. Final step: {}", self.step);

        Ok(())
    }

    fn evaluate(&mut self, eval_loader: &mut DataLoader, _device: &Device) -> Result<f64> {
        let mut total_loss = 0.0;
        let mut count = 0;

        for _ in 0..self.config.eval_steps.min(eval_loader.len()) {
            let batch = eval_loader.next_batch(self.config.micro_batch_size);
            for tokens in batch {
                // Fixed seed so eval loss is comparable across runs.
                let mut eval_rng = StdRng::seed_from_u64(12345);
                let (input_ids, labels) = corrupt(&self.glm_config, &tokens, &mut eval_rng);

                let logits = self.model.forward_causal(&input_ids)?;
                let loss = cross_entropy_loss(&logits, &labels)?;

                total_loss += loss.to_scalar::<f32>()? as f64;
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_loss / count as f64)
        } else {
            Ok(0.0)
        }
    }

    pub fn save_checkpoint(&self) -> Result<()> {
        let dir = Path::new(&self.config.checkpoint_dir);
        std::fs::create_dir_all(dir).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to create checkpoint dir: {e}"))
        })?;

        let params = self.model.param_vars();
        let names = param_names(self.glm_config.num_layers);

        // Save as safetensors
        let mut tensor_data = Vec::new();
        for (i, var) in params.iter().enumerate() {
            let name = names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
            let tensor = var.as_tensor();
            tensor_data.push((name.to_string(), tensor));
        }

        let path = dir.join(format!("model_step_{:06}.safetensors", self.step));
        save_safetensors(&path, &tensor_data)?;

        // Save optimizer state (simplified - just step count)
        if self.config.save_optimizer_state {
            let opt_path = dir.join(format!("optimizer_step_{:06}.json", self.step));
            let state = serde_json::json!({
                "step": self.step,
            });
            let opt_json = serde_json::to_string_pretty(&state).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to serialize optimizer state: {e}"))
            })?;
            std::fs::write(&opt_path, opt_json).map_err(|e| {
                candle_core::Error::Msg(format!("Failed to write optimizer state: {e}"))
            })?;
        }

        // Save training state
        let state_path = dir.join("training_state.json");
        let state = serde_json::json!({
            "step": self.step,
            "lr": self.lr_scheduler.get_lr(),
        });
        let state_json = serde_json::to_string_pretty(&state)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize state: {e}")))?;
        std::fs::write(&state_path, state_json)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write state: {e}")))?;

        println!("  Checkpoint saved to {:?} (step {})", path, self.step);
        Ok(())
    }

    pub fn load_checkpoint(&mut self) -> Result<()> {
        let dir = Path::new(&self.config.checkpoint_dir);
        if !dir.exists() {
            return Ok(());
        }

        // Find latest checkpoint
        let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("safetensors"))
            .filter(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("model_step_")
            })
            .collect();

        checkpoints.sort();

        if let Some(latest) = checkpoints.last() {
            println!("Loading checkpoint from {:?}", latest);
            load_safetensors(latest, &mut self.model)?;

            // Load optimizer state
            let opt_path = dir.join(
                latest
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .replace("model_", "optimizer_")
                    .replace(".safetensors", ".json"),
            );
            if opt_path.exists() {
                if let Ok(content) = std::fs::read_to_string(&opt_path) {
                    if let Ok(state) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(_step) = state.get("step").and_then(|v| v.as_u64()) {
                            // Note: AdamW doesn't expose step_count setter, but we track our own step
                        }
                    }
                }
            }

            // Load training state
            let state_path = dir.join("training_state.json");
            if state_path.exists() {
                let content = std::fs::read_to_string(&state_path)
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to read state: {e}")))?;
                let state: serde_json::Value = serde_json::from_str(&content)
                    .map_err(|e| candle_core::Error::Msg(format!("Failed to parse state: {e}")))?;
                self.step = state.get("step").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                self.lr_scheduler.set_step(self.step);
            }

            println!("Resumed from step {}", self.step);
        }

        Ok(())
    }

    fn cleanup_old_checkpoints(&self) -> Result<()> {
        let dir = Path::new(&self.config.checkpoint_dir);
        let mut model_checkpoints: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .starts_with("model_step_")
            })
            .collect();

        model_checkpoints.sort();

        while model_checkpoints.len() > self.config.keep_last_n_checkpoints {
            let oldest = model_checkpoints.remove(0);
            let _ = std::fs::remove_file(&oldest);
            let opt_file = dir.join(
                oldest
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .replace("model_", "optimizer_")
                    .replace(".safetensors", ".json"),
            );
            let _ = std::fs::remove_file(&opt_file);
        }

        Ok(())
    }

    pub fn step(&self) -> usize {
        self.step
    }
}

/// Scale every gradient so the global L2 norm across `params` is at most
/// `max_norm`. Returns the norm before clipping.
pub fn grad_global_norm(grads: &GradStore, params: &[Var]) -> Result<f64> {
    let mut sum_sq = 0f64;
    for var in params {
        if let Some(grad) = grads.get(var) {
            sum_sq += grad
                .sqr()?
                .sum_all()?
                .to_dtype(DType::F32)?
                .to_scalar::<f32>()? as f64;
        }
    }
    Ok(sum_sq.sqrt())
}

/// Clip gradients in place by global norm. Returns the norm before clipping.
pub fn clip_grad_norm(grads: &mut GradStore, params: &[Var], max_norm: f64) -> Result<f64> {
    let norm = grad_global_norm(grads, params)?;
    if norm.is_finite() && norm > max_norm {
        let scale = max_norm / norm;
        for var in params {
            let scaled = grads.get(var).map(|grad| grad * scale).transpose()?;
            if let Some(scaled) = scaled {
                grads.insert(var, scaled);
            }
        }
    }
    Ok(norm)
}

fn param_names(num_layers: usize) -> Vec<String> {
    let mut names = Vec::new();
    names.push("embedding.weight".to_string());
    names.push("pos_1_embedding".to_string());
    names.push("pos_2_embedding".to_string());
    for i in 0..num_layers {
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

/// Flatten a tensor into safetensors bytes.
///
/// The previous version called `tensor.to_vec1::<u8>()`, which fails on anything
/// of rank > 1 — so every checkpoint save errored with
/// "unexpected rank, expected: 1, got: 2" the moment it reached a weight matrix.
pub fn tensor_to_bytes(tensor: &Tensor) -> Result<(Vec<u8>, safetensors::Dtype)> {
    let flat = tensor.flatten_all()?;
    match tensor.dtype() {
        DType::F32 => {
            let values: Vec<f32> = flat.to_vec1()?;
            Ok((
                bytemuck::cast_slice(&values).to_vec(),
                safetensors::Dtype::F32,
            ))
        }
        DType::F16 => {
            let values: Vec<half::f16> = flat.to_vec1()?;
            Ok((
                bytemuck::cast_slice(&values).to_vec(),
                safetensors::Dtype::F16,
            ))
        }
        DType::BF16 => {
            let values: Vec<half::bf16> = flat.to_vec1()?;
            Ok((
                bytemuck::cast_slice(&values).to_vec(),
                safetensors::Dtype::BF16,
            ))
        }
        dtype => Err(candle_core::Error::Msg(format!(
            "unsupported checkpoint dtype: {dtype:?}"
        ))),
    }
}

fn save_safetensors(path: &Path, tensors: &[(String, &Tensor)]) -> Result<()> {
    use safetensors::serialize;
    use std::collections::HashMap;

    // Collect tensor data first to keep it alive
    let mut tensor_data = Vec::new();

    for (name, tensor) in tensors {
        let (data, st_dtype) = tensor_to_bytes(tensor)?;
        let shape = tensor.shape().dims().to_vec();
        tensor_data.push((name.clone(), data, shape, st_dtype));
    }

    let mut tensor_map = HashMap::new();
    for (name, data, shape, st_dtype) in &tensor_data {
        tensor_map.insert(
            name.clone(),
            TensorView::new(*st_dtype, shape.clone(), data)?,
        );
    }

    let bytes = serialize(tensor_map, &None)?;
    std::fs::write(path, bytes)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write safetensors: {e}")))?;

    Ok(())
}

fn load_safetensors(path: &Path, model: &mut TrainableGLMModel) -> Result<()> {
    let data = std::fs::read(path)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read safetensors: {e}")))?;

    let safe = SafeTensors::deserialize(&data)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize: {e}")))?;

    let params = model.param_vars();
    let names = param_names(model.num_layers);

    for (i, var) in params.iter().enumerate() {
        let name = names.get(i).map(|s| s.as_str()).unwrap_or("unknown");
        let tensor_view = safe.tensor(name).map_err(|e| {
            candle_core::Error::Msg(format!("Failed to get tensor {}: {}", name, e))
        })?;
        let shape = tensor_view.shape();
        let tensor_data = tensor_view.data();
        let dtype = tensor_view.dtype();

        let tensor = match dtype {
            safetensors::Dtype::F32 => {
                let vec: Vec<f32> = bytemuck::cast_slice(tensor_data).to_vec();
                Tensor::from_vec(vec, shape, var.as_tensor().device())?
            }
            safetensors::Dtype::F16 => {
                let vec: Vec<half::f16> = bytemuck::cast_slice(tensor_data).to_vec();
                Tensor::from_vec(vec, shape, var.as_tensor().device())?
            }
            safetensors::Dtype::BF16 => {
                let vec: Vec<half::bf16> = bytemuck::cast_slice(tensor_data).to_vec();
                Tensor::from_vec(vec, shape, var.as_tensor().device())?
            }
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unsupported dtype: {:?}",
                    dtype
                )))
            }
        };

        var.set(&tensor)?;
    }

    Ok(())
}

fn load_data(data_dir: &Path, tokenizer: &CodeGenTokenizer) -> Result<Vec<DataTrainingExample>> {
    let mut examples = Vec::new();

    if !data_dir.exists() {
        return Ok(examples);
    }

    let entries = std::fs::read_dir(data_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read data dir: {e}")))?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("py") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(ids) = tokenizer.encode(&content) {
                    if ids.len() > 10 {
                        examples.push(DataTrainingExample { tokens: ids });
                    }
                }
            }
        }
    }

    Ok(examples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::loss::cross_entropy_loss;

    #[test]
    fn clipping_bounds_the_global_norm() -> Result<()> {
        let device = Device::Cpu;
        let config = GLMConfig {
            vocab_size: 64,
            hidden_dim: 32,
            num_layers: 2,
            num_heads: 4,
            ffn_dim: 64,
            max_seq_len: 16,
            ..Default::default()
        };
        let model = TrainableGLMModel::new(config, &device)?;
        let params = model.param_vars();

        let logits = model.forward_causal(&[3u32, 9, 14, 2, 41])?;
        let loss = cross_entropy_loss(&logits, &[1i64, 2, 3, 4, 5])?;
        let mut grads = loss.backward()?;

        let before = grad_global_norm(&grads, &params)?;
        assert!(before > 0.0, "no gradients to clip");

        let max_norm = before / 10.0;
        let reported = clip_grad_norm(&mut grads, &params, max_norm)?;
        assert!((reported - before).abs() < 1e-6);

        let after = grad_global_norm(&grads, &params)?;
        assert!(
            (after - max_norm).abs() < 1e-4,
            "clipped norm {after} should sit at the bound {max_norm}"
        );
        Ok(())
    }

    #[test]
    fn clipping_leaves_small_gradients_alone() -> Result<()> {
        let device = Device::Cpu;
        let config = GLMConfig {
            vocab_size: 64,
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            ffn_dim: 64,
            max_seq_len: 16,
            ..Default::default()
        };
        let model = TrainableGLMModel::new(config, &device)?;
        let params = model.param_vars();

        let logits = model.forward_causal(&[3u32, 9, 14])?;
        let loss = cross_entropy_loss(&logits, &[1i64, 2, 3])?;
        let mut grads = loss.backward()?;

        let before = grad_global_norm(&grads, &params)?;
        clip_grad_norm(&mut grads, &params, before * 10.0)?;
        let after = grad_global_norm(&grads, &params)?;

        assert!((after - before).abs() < 1e-6, "{before} changed to {after}");
        Ok(())
    }
}
