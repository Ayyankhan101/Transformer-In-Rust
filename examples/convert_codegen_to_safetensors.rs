//! CodeGen PyTorch .bin → safetensors Converter
//!
//! Reads a HuggingFace PyTorch checkpoint (model-*.bin) for
//! Salesforce/codegen-350M-multi and writes it as a safetensors file
//! that can be loaded directly without the PyTorch pickle dependency.
//!
//! Usage:
//!   cargo run --example convert_codegen_to_safetensors -- \
//!       /path/to/codegen_weights/pytorch_model.bin \
//!       /path/to/codegen_weights/model.safetensors
//!
//! The result is picked up automatically: `ModelContext` prefers
//! `model.safetensors` over `pytorch_model.bin` in the weights directory.
//!
//!   cargo run --release -- --weights-dir /path/to/codegen_weights complete "def f():"

use std::fs;
use std::path::PathBuf;

use candle_core::pickle::PthTensors;
use candle_core::{DType, Device};

const SUPPORTED_DTYPES: &[&str] = &["F32", "F16", "BF16"];

fn parse_dtype(s: &str) -> Option<DType> {
    match s.to_uppercase().as_str() {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        _ => None,
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "Usage: convert_codegen_to_safetensors <input.bin> <output.safetensors> [--dtype F32|F16|BF16]";

    if args.len() < 3 {
        eprintln!("{usage}");
        std::process::exit(1);
    }

    let input_path = PathBuf::from(&args[1]);
    let output_path = PathBuf::from(&args[2]);

    let dtype = if args.len() > 3 && args[3] == "--dtype" {
        if let Some(d) = args.get(4).and_then(|s| parse_dtype(s)) {
            d
        } else {
            eprintln!(
                "Error: unsupported dtype. Choose from: {}",
                SUPPORTED_DTYPES.join(", ")
            );
            std::process::exit(1);
        }
    } else {
        DType::F32
    };

    if !input_path.exists() {
        eprintln!("Error: input file not found: {}", input_path.display());
        std::process::exit(1);
    }

    println!("=== CodeGen PyTorch → Safetensors Converter ===");
    println!("Input:  {}", input_path.display());
    println!("Output: {}", output_path.display());
    println!("Dtype:  {:?}", dtype);
    println!();

    let device = Device::Cpu;
    let pth = PthTensors::new(&input_path, None)?;

    println!("Loading tensors from PyTorch checkpoint...");

    // Collect all tensor names from the pickle file
    // We iterate over the expected CodeGen weight names
    let weight_names = collect_codegen_weight_names(&pth)?;

    if weight_names.is_empty() {
        // Fallback: try to enumerate by known patterns
        eprintln!("Warning: Could not find any tensors matching CodeGen patterns.");
        eprintln!("Trying generic enumeration...");
        return enumerate_and_save(&pth, &output_path, dtype, &device);
    }

    use std::collections::HashMap;

    // Store data with metadata; TensorView will borrow from this
    struct StoredTensor {
        name: String,
        data: Vec<u8>,
        shape: Vec<usize>,
        st_dtype: safetensors::Dtype,
    }

    let mut stored: Vec<StoredTensor> = Vec::new();

    for name in &weight_names {
        println!("  Converting: {name}");

        let tensor = pth
            .get(name)?
            .ok_or_else(|| anyhow::anyhow!("Tensor '{name}' not found"))?;

        // Convert to target dtype
        let tensor = if tensor.dtype() != dtype {
            tensor.to_dtype(dtype)?.to_device(&device)?
        } else {
            tensor.to_device(&device)?
        };

        let shape: Vec<usize> = tensor.dims().to_vec();
        let flat = tensor.flatten_all()?;
        let data = flat.to_vec1::<f32>()?;
        let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

        let st_dtype = match dtype {
            DType::F32 => safetensors::Dtype::F32,
            DType::F16 => safetensors::Dtype::F16,
            DType::BF16 => safetensors::Dtype::BF16,
            _ => safetensors::Dtype::F32,
        };

        stored.push(StoredTensor {
            name: name.clone(),
            data: bytes,
            shape,
            st_dtype,
        });
    }

    // Serialize to safetensors
    println!("\nSerializing {} tensors to safetensors...", stored.len());

    let mut tensor_map = HashMap::new();
    for s in &stored {
        let view = safetensors::tensor::TensorView::new(s.st_dtype, s.shape.clone(), &s.data)
            .map_err(|e| anyhow::anyhow!("Failed to create TensorView: {e}"))?;
        tensor_map.insert(s.name.clone(), view);
    }

    let serialized = safetensors::serialize(tensor_map, &None)
        .map_err(|e| anyhow::anyhow!("Failed to serialize safetensors: {e}"))?;

    // Write to file
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, &serialized)?;

    let file_size_mb = serialized.len() as f64 / (1024.0 * 1024.0);
    println!("\n✓ Successfully converted!");
    println!("  Output: {}", output_path.display());
    println!("  Size:   {:.2} MB", file_size_mb);

    Ok(())
}

/// Collect tensor names following the CodeGen-350M naming convention.
fn collect_codegen_weight_names(pth: &PthTensors) -> anyhow::Result<Vec<String>> {
    let _expected_weights = [
        "transformer.wte.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
        "lm_head.bias",
    ];

    // Try to detect number of layers
    let num_layers = (0..100)
        .find(|&i| {
            pth.get(&format!("transformer.h.{i}.ln_1.weight"))
                .ok()
                .flatten()
                .is_none()
        })
        .unwrap_or(0);

    if num_layers == 0 {
        // Check at least layer 0 exists
        if pth
            .get("transformer.h.0.ln_1.weight")
            .ok()
            .flatten()
            .is_none()
        {
            return Ok(Vec::new());
        }
    }

    // We'll just try all known patterns and include what exists
    let patterns = [
        "transformer.wte.weight",
        "transformer.ln_f.weight",
        "transformer.ln_f.bias",
        "lm_head.weight",
        "lm_head.bias",
    ];

    let layer_patterns = [
        "ln_1.weight",
        "ln_1.bias",
        "attn.qkv_proj.weight",
        "attn.out_proj.weight",
        "ln_2.weight",
        "ln_2.bias",
        "mlp.fc_in.weight",
        "mlp.fc_in.bias",
        "mlp.fc_out.weight",
        "mlp.fc_out.bias",
    ];

    let mut names = Vec::new();

    // Check patterns at root level
    for pattern in &patterns {
        if pth.get(pattern).ok().flatten().is_some() {
            names.push(pattern.to_string());
        }
    }

    // Check layer patterns
    for layer in 0..num_layers {
        for pattern in &layer_patterns {
            let name = format!("transformer.h.{layer}.{pattern}");
            if pth.get(&name).ok().flatten().is_some() {
                names.push(name);
            }
        }
    }

    Ok(names)
}

/// Fallback: enumerate all tensors in the pickle file
fn enumerate_and_save(
    pth: &PthTensors,
    output_path: &PathBuf,
    dtype: DType,
    device: &Device,
) -> anyhow::Result<()> {
    let all_patterns = generate_all_codegen_patterns(24); // max 24 layers

    // Store data with matching shape/dtype info; TensorView will borrow from this Vec
    struct StoredTensor {
        name: String,
        data: Vec<u8>,
        shape: Vec<usize>,
        st_dtype: safetensors::Dtype,
    }

    let mut stored: Vec<StoredTensor> = Vec::new();

    for name in &all_patterns {
        if let Ok(Some(tensor)) = pth.get(name) {
            println!("  Found: {name}");

            let tensor = if tensor.dtype() != dtype {
                tensor.to_dtype(dtype)?.to_device(device)?
            } else {
                tensor.to_device(device)?
            };

            let shape: Vec<usize> = tensor.dims().to_vec();
            let flat = tensor.flatten_all()?;
            let data = flat.to_vec1::<f32>()?;
            let bytes: Vec<u8> = bytemuck::cast_slice(&data).to_vec();

            let st_dtype = match dtype {
                DType::F32 => safetensors::Dtype::F32,
                DType::F16 => safetensors::Dtype::F16,
                DType::BF16 => safetensors::Dtype::BF16,
                _ => safetensors::Dtype::F32,
            };

            stored.push(StoredTensor {
                name: name.clone(),
                data: bytes,
                shape,
                st_dtype,
            });
        }
    }

    if stored.is_empty() {
        anyhow::bail!("No tensors found in the pickle file. Is this a CodeGen checkpoint?");
    }

    use std::collections::HashMap;
    let mut tensor_map = HashMap::new();
    for s in &stored {
        let view = safetensors::tensor::TensorView::new(s.st_dtype, s.shape.clone(), &s.data)
            .map_err(|e| anyhow::anyhow!("Failed to create TensorView: {e}"))?;
        tensor_map.insert(s.name.clone(), view);
    }

    let serialized = safetensors::serialize(tensor_map, &None)
        .map_err(|e| anyhow::anyhow!("Failed to serialize safetensors: {e}"))?;

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(output_path, &serialized)?;

    let file_size_mb = serialized.len() as f64 / (1024.0 * 1024.0);
    println!("\n✓ Successfully converted {} tensors!", stored.len());
    println!("  Output: {}", output_path.display());
    println!("  Size:   {:.2} MB", file_size_mb);

    Ok(())
}

/// Generate all expected CodeGen weight names for up to N layers
fn generate_all_codegen_patterns(max_layers: usize) -> Vec<String> {
    // Root-level weights
    let mut names = vec![
        "transformer.wte.weight".to_string(),
        "transformer.ln_f.weight".to_string(),
        "transformer.ln_f.bias".to_string(),
        "lm_head.weight".to_string(),
        "lm_head.bias".to_string(),
    ];

    // Per-layer weights
    for i in 0..max_layers {
        names.push(format!("transformer.h.{i}.ln_1.weight"));
        names.push(format!("transformer.h.{i}.ln_1.bias"));
        names.push(format!("transformer.h.{i}.attn.qkv_proj.weight"));
        names.push(format!("transformer.h.{i}.attn.out_proj.weight"));
        names.push(format!("transformer.h.{i}.ln_2.weight"));
        names.push(format!("transformer.h.{i}.ln_2.bias"));
        names.push(format!("transformer.h.{i}.mlp.fc_in.weight"));
        names.push(format!("transformer.h.{i}.mlp.fc_in.bias"));
        names.push(format!("transformer.h.{i}.mlp.fc_out.weight"));
        names.push(format!("transformer.h.{i}.mlp.fc_out.bias"));
    }

    names
}
