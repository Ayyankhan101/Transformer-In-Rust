use std::path::{Path, PathBuf};
use candle_core::{Device, Result, Tensor, DType};
use candle_nn::Var;
use safetensors::tensor::TensorView;
use safetensors::serialize;

pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    save_optimizer_state: bool,
    keep_last_n: usize,
}

impl CheckpointManager {
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P, save_optimizer_state: bool, keep_last_n: usize) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            save_optimizer_state,
            keep_last_n,
        }
    }

    pub fn save_model(
        &self,
        step: usize,
        model_vars: &[(&str, &Var)],
    ) -> Result<()> {
        std::fs::create_dir_all(&self.checkpoint_dir)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to create checkpoint dir: {e}")))?;

        let model_path = self.checkpoint_dir.join(format!("model_step_{:07}.safetensors", step));
        let mut tensors = Vec::new();

        for (name, var) in model_vars {
            let tensor = var.as_tensor();
            let tensor_view = TensorView::new(tensor.dtype(), tensor.shape(), tensor.as_slice()?)?;
            tensors.push((name.to_string(), tensor_view));
        }

        let bytes = serialize(&tensors, &None)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize: {e}")))?;

        std::fs::write(&model_path, bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write model: {e}")))?;

        // Save step metadata
        let step_path = self.checkpoint_dir.join(format!("step_{:07}.txt", step));
        std::fs::write(&step_path, step.to_string())
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write step: {e}")))?;

        // Clean old checkpoints
        self.cleanup_old_checkpoints(step)?;

        println!("  Checkpoint saved: {:?} (step {})", model_path, step);
        Ok(())
    }

    pub fn save_optimizer_state(
        &self,
        step: usize,
        optimizer_state: &[(&str, Vec<Tensor>)],
    ) -> Result<()> {
        if !self.save_optimizer_state {
            return Ok(());
        }

        let opt_path = self.checkpoint_dir.join(format!("optimizer_step_{:07}.safetensors", step));
        let mut tensors = Vec::new();

        for (name, state_tensors) in optimizer_state {
            for (i, tensor) in state_tensors.iter().enumerate() {
                let tensor_name = format!("{}_{}", name, i);
                let tensor_view = TensorView::new(tensor.dtype(), tensor.shape(), tensor.as_slice()?)?;
                tensors.push((tensor_name, tensor_view));
            }
        }

        let bytes = serialize(&tensors, &None)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to serialize optimizer: {e}")))?;

        std::fs::write(&opt_path, bytes)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to write optimizer: {e}")))?;

        Ok(())
    }

    pub fn load_model<P: AsRef<Path>>(
        &self,
        path: P,
        model_vars: &mut [(&str, &mut Var)],
        device: &Device,
    ) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read checkpoint: {e}")))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize: {e}")))?;

        for (name, var) in model_vars {
            if let Ok(tensor_view) = tensors.tensor(name) {
                let tensor = Tensor::from_slice(
                    tensor_view.data(),
                    tensor_view.shape(),
                    device,
                )?;
                var.set(&tensor)?;
            }
        }

        Ok(())
    }

    pub fn load_optimizer_state<P: AsRef<Path>>(
        &self,
        path: P,
        optimizer_vars: &mut [(&str, &mut [Tensor])],
    ) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read optimizer: {e}")))?;

        let tensors = safetensors::SafeTensors::deserialize(&data)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to deserialize optimizer: {e}")))?;

        for (prefix, var_tensors) in optimizer_vars {
            for (i, var) in var_tensors.iter_mut().enumerate() {
                let name = format!("{}_{}", prefix, i);
                if let Ok(tensor_view) = tensors.tensor(&name) {
                    let tensor = Tensor::from_slice(
                        tensor_view.data(),
                        tensor_view.shape(),
                        var.device(),
                    )?;
                    *var = tensor;
                }
            }
        }

        Ok(())
    }

    pub fn find_latest_checkpoint(&self) -> Option<(usize, PathBuf)> {
        let mut checkpoints = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("model_step_") && name.ends_with(".safetensors") {
                        if let Some(step_str) = name.strip_prefix("model_step_").and_then(|s| s.strip_suffix(".safetensors")) {
                            if let Ok(step) = step_str.parse::<usize>() {
                                checkpoints.push((step, path));
                            }
                        }
                    }
                }
            }
        }

        checkpoints.sort_by_key(|&(step, _)| step);
        checkpoints.into_iter().next_back()
    }

    fn cleanup_old_checkpoints(&self, current_step: usize) -> Result<()> {
        let mut checkpoints = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("model_step_") && name.ends_with(".safetensors") {
                        if let Some(step_str) = name.strip_prefix("model_step_").and_then(|s| s.strip_suffix(".safetensors")) {
                            if let Ok(step) = step_str.parse::<usize>() {
                                checkpoints.push((step, path));
                            }
                        }
                    }
                }
            }
        }

        checkpoints.sort_by_key(|&(step, _)| step);

        if checkpoints.len() > self.keep_last_n {
            for (step, path) in checkpoints.iter().take(checkpoints.len() - self.keep_last_n) {
                let _ = std::fs::remove_file(path);
                // Also remove corresponding step file and optimizer file
                let _ = std::fs::remove_file(self.checkpoint_dir.join(format!("step_{:07}.txt", step)));
                let _ = std::fs::remove_file(self.checkpoint_dir.join(format!("optimizer_step_{:07}.safetensors", step)));
            }
        }

        Ok(())
    }
}