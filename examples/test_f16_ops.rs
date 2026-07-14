use candle_core::{Device, Tensor, DType};

fn main() -> candle_core::Result<()> {
    let device = Device::Cpu;

    let a = Tensor::randn(0.0f32, 1.0f32, (2, 8, 4), &device)?;
    let a_f16 = a.to_dtype(DType::F16)?;

    let sm = candle_nn::ops::softmax(&a_f16, 2)?;
    println!("Softmax shape: {:?}, dtype: {:?}", sm.shape(), sm.dtype());

    let b_f16 = a_f16.add(&a_f16)?;
    println!("Add shape: {:?}, dtype: {:?}", b_f16.shape(), b_f16.dtype());

    let permuted = a_f16.permute((0, 2, 1))?;
    println!("Permute shape: {:?}, dtype: {:?}", permuted.shape(), permuted.dtype());

    let t = a_f16.transpose(1, 2)?;
    println!("Transpose shape: {:?}, dtype: {:?}", t.shape(), t.dtype());

    let v = Tensor::randn(0.0f32, 1.0f32, (2, 4, 8), &device)?;
    let v_f16 = v.to_dtype(DType::F16)?;
    let attn = sm.broadcast_matmul(&v_f16)?;
    println!("Matmul shape: {:?}, dtype: {:?}", attn.shape(), attn.dtype());

    let residual = Tensor::randn(0.0f32, 1.0f32, (2, 8, 4), &device)?;
    let residual_f16 = residual.to_dtype(DType::F16)?;
    let out = attn.broadcast_add(&residual_f16)?;
    println!("Broadcast add shape: {:?}, dtype: {:?}", out.shape(), out.dtype());

    let reshaped = out.reshape((16, 4))?;
    println!("Reshape shape: {:?}, dtype: {:?}", reshaped.shape(), reshaped.dtype());

    let indexed = out.get(0)?;
    println!("Index shape: {:?}, dtype: {:?}", indexed.shape(), indexed.dtype());

    let to_vec = out.get(0)?.get(1)?.to_vec1::<f32>()?;
    println!("to_vec1 (as f32) len: {}", to_vec.len());

    let to_scalar = out.get(0)?.get(0)?.get(0)?.to_scalar::<f32>()?;
    println!("to_scalar value: {}", to_scalar);

    let argmax = out.argmax(2)?;
    println!("Argmax shape: {:?}, dtype: {:?}", argmax.shape(), argmax.dtype());

    println!("\nAll F16 ops work on CPU!");
    Ok(())
}
