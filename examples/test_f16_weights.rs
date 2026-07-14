use candle_core::{Device, DType, Tensor};

fn load_tensor(pth: &candle_core::pickle::PthTensors, name: &str, dtype: DType, device: &Device) -> Option<Tensor> {
    let t = pth.get(name).ok()?;
    let t = t?;
    let t = if t.dtype() != dtype { t.to_dtype(dtype).ok()? } else { t };
    t.to_device(device).ok()
}

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let pth = candle_core::pickle::PthTensors::new("codegen_weights/pytorch_model.bin", None)?;

    // Check weight dtypes
    let w = pth.get("token_embd.weight").unwrap().unwrap();
    println!("token_embd.weight: dtype={:?}, shape={:?}", w.dtype(), w.shape());

    let w = pth.get("transformer.h.0.attn.qkv_proj.weight").unwrap().unwrap();
    println!("qkv_proj.weight: dtype={:?}, shape={:?}", w.dtype(), w.shape());

    let w = pth.get("transformer.h.0.ln_1.weight").unwrap().unwrap();
    println!("ln_1.weight: dtype={:?}, shape={:?}", w.dtype(), w.shape());

    let w = pth.get("lm_head.weight").unwrap().unwrap();
    println!("lm_head.weight: dtype={:?}, shape={:?}", w.dtype(), w.shape());

    // Test F16 matmul
    let a = Tensor::randn(0.0f32, 1.0f32, (1, 1024), &device)?.to_dtype(DType::F16)?;
    let b = Tensor::zeros((1024, 3072), DType::F16, &device)?;
    let c = a.broadcast_matmul(&b.unsqueeze(0)?)?;
    println!("F16 matmul works: shape={:?}", c.shape());

    // Test what happens when we load as F16
    let _w_f16 = load_tensor(&pth, "token_embd.weight", DType::F16, &device).unwrap();
    println!("Loaded token_embd as F16");

    let _w_f32 = load_tensor(&pth, "token_embd.weight", DType::F32, &device).unwrap();
    println!("Loaded token_embd as F32");

    // Test F16*F32 matmul
    let a_f16 = Tensor::randn(0.0f32, 1.0f32, (1, 1024), &device)?.to_dtype(DType::F16)?;
    let b_f32 = Tensor::zeros((1024, 3072), DType::F32, &device)?;
    let r = a_f16.broadcast_matmul(&b_f32.unsqueeze(0)?);
    println!("F16*F32 matmul result: {:?}", r);

    Ok(())
}
