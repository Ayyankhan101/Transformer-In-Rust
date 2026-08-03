use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use bytemuck;
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use half;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;

use crate::glm::config::GLMConfig;
use crate::glm::trainable::TrainableGLMModel;
use crate::tokenizer::CodeGenTokenizer;
use crate::training::config::{TrainConfig, TrainingConfig};
use crate::training::data::{
    download_default_data, split_train_eval, DataLoader, TrainingExample as DataTrainingExample,
};
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
        })
    }

    pub fn train_step(&mut self, token_ids: &[u32], _device: &Device) -> Result<f64> {
        let n = token_ids.len();
        let mask_token_id = self.glm_config.vocab_size as u32 - 1;

        let mut rng_state = self.step as u64;
        let mut rand = || -> f64 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (rng_state as f64) / (u64::MAX as f64)
        };

        let mut input_ids = token_ids.to_vec();
        let mut labels = vec![-1i64; n];

        for i in 0..n {
            let r = rand();
            if r < self.glm_config.blank_ratio {
                labels[i] = token_ids[i] as i64;
                if rand() < self.glm_config.mask_ratio {
                    input_ids[i] = mask_token_id;
                } else {
                    input_ids[i] = (rand() * self.glm_config.vocab_size as f64) as u32;
                }
            }
        }

        let logits = self.model.forward_causal(&input_ids)?;
        let loss = cross_entropy_loss(&logits, &labels)?;

        let loss_scalar = loss.to_scalar::<f32>()? as f64;

        // backward_step does backward + step in one call
        self.optimizer.backward_step(&loss)?;

        // Update learning rate for next step
        let current_lr = self.lr_scheduler.get_lr();
        self.optimizer.set_learning_rate(current_lr);

        self.step += 1;
        self.lr_scheduler.step();

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
            let batch = train_loader.next_batch(self.config.micro_batch_size);

            for tokens in batch {
                let loss = self.train_step(&tokens, device)?;

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
                let n = tokens.len();
                let mask_token_id = self.glm_config.vocab_size as u32 - 1;

                let mut input_ids = tokens.clone();
                let mut labels = vec![-1i64; n];

                // Use fixed seed for reproducible eval
                let mut rng_state = 12345 + self.step as u64;
                let mut rand = || -> f64 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (rng_state as f64) / (u64::MAX as f64)
                };

                for i in 0..n {
                    if rand() < self.glm_config.blank_ratio {
                        labels[i] = tokens[i] as i64;
                        if rand() < self.glm_config.mask_ratio {
                            input_ids[i] = mask_token_id;
                        } else {
                            input_ids[i] = (rand() * self.glm_config.vocab_size as f64) as u32;
                        }
                    }
                }

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

fn cross_entropy_loss(logits: &Tensor, labels: &[i64]) -> Result<Tensor> {
    let seq_len = logits.dims()[1];
    let mut total_loss = 0.0f32;
    let mut count = 0usize;

    for (i, &label) in labels.iter().enumerate().take(seq_len) {
        if label >= 0 {
            let logits_i = logits.get(0)?.get(i)?;
            let ce = candle_nn::ops::log_softmax(&logits_i, 0)?
                .get(label as usize)?
                .neg()?;
            total_loss += ce.to_scalar::<f32>()?;
            count += 1;
        }
    }

    if count > 0 {
        total_loss /= count as f32;
    }

    Tensor::new(total_loss, logits.device())
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

fn save_safetensors(path: &Path, tensors: &[(String, &Tensor)]) -> Result<()> {
    use safetensors::serialize;
    use std::collections::HashMap;

    // Collect tensor data first to keep it alive
    let mut tensor_data = Vec::new();

    for (name, tensor) in tensors {
        let data = tensor.to_vec1::<u8>()?;
        let shape = tensor.shape().dims().to_vec();
        let dtype = tensor.dtype();
        let st_dtype = match dtype {
            DType::F32 => safetensors::Dtype::F32,
            DType::F16 => safetensors::Dtype::F16,
            DType::BF16 => safetensors::Dtype::BF16,
            _ => {
                return Err(candle_core::Error::Msg(format!(
                    "Unsupported dtype: {:?}",
                    dtype
                )))
            }
        };
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
