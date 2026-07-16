use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;
    let a = Tensor::randn(0.0f32, 1.0f32, (64, 64), &device)?;
    let b = Tensor::randn(0.0f32, 1.0f32, (64, 64), &device)?;
    let a_f16 = a.to_dtype(DType::F16)?;
    let b_f16 = b.to_dtype(DType::F16)?;
    let c = a_f16.matmul(&b_f16)?;
    println!("F16 matmul result shape: {:?}", c.shape());
    println!("OK: F16 matmul works on CPU");
    Ok(())
}
