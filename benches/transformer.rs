use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use candle_core::{Device, Tensor, DType};

pub fn bench_attention(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("attention");

    for &hidden_dim in &[256, 512, 1024] {
        for &num_heads in &[4, 8, 16] {
            if hidden_dim % num_heads != 0 {
                continue;
            }
            let head_dim = hidden_dim / num_heads;
            let seq_len = 64;

            let qkv_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), &device).unwrap();
            let out_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device).unwrap();

            group.throughput(Throughput::Elements((seq_len * hidden_dim * num_heads) as u64));
            group.bench_with_input(
                BenchmarkId::new("f32", format!("h{}_nh{}", hidden_dim, num_heads)),
                &(&qkv_weight, &out_weight, hidden_dim, num_heads, head_dim, seq_len),
                |b, (qkv_weight, out_weight, hidden_dim, num_heads, head_dim, seq_len)| {
                    b.iter(|| {
                        let x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap();
                        let qkv = x.broadcast_matmul(&qkv_weight.unsqueeze(0).unwrap()).unwrap();
                        let qkv = qkv.reshape((1, *seq_len, 3, *num_heads, *head_dim)).unwrap();
                        let qkv = qkv.permute((0, 3, 2, 1, 4)).unwrap();
                        let q = qkv.get_on_dim(2, 0).unwrap();
                        let v = qkv.get_on_dim(2, 1).unwrap();
                        let k = qkv.get_on_dim(2, 2).unwrap();

                        let scale = 1.0 / (*head_dim as f64).sqrt();
                        let scores = q.broadcast_matmul(&k.transpose(2, 3).unwrap()).unwrap();
                        let scores = (scores * scale).unwrap();

                        let weights = candle_nn::ops::softmax(&scores, 3).unwrap();
                        let context = weights.broadcast_matmul(&v).unwrap();
                        let context = context.permute((0, 2, 1, 3)).unwrap()
                            .reshape((1, *seq_len, *hidden_dim)).unwrap();
                        context.broadcast_matmul(&out_weight.unsqueeze(0).unwrap())
                    });
                },
            );
        }
    }
    group.finish();
}

pub fn bench_ffn(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("ffn");

    for &hidden_dim in &[256, 512, 1024] {
        for &ffn_dim in &[1024, 2048, 4096] {
            let seq_len = 64;

            let fc_in = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), &device).unwrap();
            let fc_out = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), &device).unwrap();

            group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
            group.bench_with_input(
                BenchmarkId::new("gelu", format!("h{}_ffn{}", hidden_dim, ffn_dim)),
                &(&fc_in, &fc_out, hidden_dim, ffn_dim, seq_len),
                |b, (fc_in, fc_out, hidden_dim, ffn_dim, seq_len)| {
                    b.iter(|| {
                        let x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap();
                        let hidden = x.broadcast_matmul(&fc_in.unsqueeze(0).unwrap()).unwrap();
                        let activated = hidden.gelu().unwrap();
                        activated.broadcast_matmul(&fc_out.unsqueeze(0).unwrap())
                    });
                },
            );
        }
    }
    group.finish();
}

pub fn bench_layernorm(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("layernorm");

    for &hidden_dim in &[256, 512, 1024] {
        let seq_len = 64;
        let weight = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
        let bias = Tensor::zeros(hidden_dim, DType::F32, &device).unwrap();
        let eps = 1e-5;

        group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
        group.bench_with_input(
            BenchmarkId::new("f32", format!("h{}", hidden_dim)),
            &(&weight, &bias, hidden_dim, seq_len, eps),
            |b, (weight, bias, hidden_dim, seq_len, eps)| {
                b.iter(|| {
                    let x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap();
                    let last_dim = x.dims().len() - 1;
                    let mean = x.mean(last_dim).unwrap();
                    let mean = mean.unsqueeze(last_dim).unwrap();
                    let x_centered = x.broadcast_sub(&mean).unwrap();
                    let variance = x_centered.sqr().unwrap().mean(last_dim).unwrap();
                    let std = (variance + *eps).unwrap().sqrt().unwrap();
                    let std = std.unsqueeze(last_dim).unwrap();
                    let normalized = x_centered.broadcast_div(&std).unwrap();
                    let weight = if weight.dtype() != normalized.dtype() {
                        weight.to_dtype(normalized.dtype()).unwrap()
                    } else {
                        weight.clone().clone()
                    };
                    let bias = if bias.dtype() != normalized.dtype() {
                        bias.to_dtype(normalized.dtype()).unwrap()
                    } else {
                        bias.clone().clone()
                    };
                    normalized.broadcast_mul(&weight).unwrap().broadcast_add(&bias).unwrap()
                });
            },
        );
    }
    group.finish();
}

pub fn bench_full_block(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("full_block");

    for &hidden_dim in &[256, 512, 1024] {
        for &num_layers in &[1, 6, 12] {
            let seq_len = 64;
            let num_heads = 8;
            let head_dim = hidden_dim / num_heads;
            let ffn_dim = hidden_dim * 4;

            let mut layers = Vec::new();
            for _ in 0..num_layers {
                let qkv_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), &device).unwrap();
                let out_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device).unwrap();
                let ln1_w = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
                let ln1_b = Tensor::zeros(hidden_dim, DType::F32, &device).unwrap();
                let ln2_w = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
                let ln2_b = Tensor::zeros(hidden_dim, DType::F32, &device).unwrap();
                let fc_in = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), &device).unwrap();
                let fc_out = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), &device).unwrap();
                layers.push((qkv_weight, out_weight, ln1_w, ln1_b, ln2_w, ln2_b, fc_in, fc_out));
            }

            group.throughput(Throughput::Elements((seq_len * hidden_dim) as u64));
            group.bench_with_input(
                BenchmarkId::new("f32", format!("h{}_l{}", hidden_dim, num_layers)),
                &(&layers, hidden_dim, num_layers, num_heads, head_dim, ffn_dim, seq_len),
                |b, (layers, hidden_dim, num_layers, num_heads, head_dim, ffn_dim, seq_len)| {
                    b.iter(|| {
                        let mut x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap();
                        for layer in layers.iter().take(*num_layers) {
                            let (qkv_weight, out_weight, ln1_w, ln1_b, ln2_w, ln2_b, fc_in, fc_out) = layer;

                            // LN1
                            let last_dim = x.dims().len() - 1;
                            let mean = x.mean(last_dim).unwrap();
                            let mean = mean.unsqueeze(last_dim).unwrap();
                            let x_centered = x.broadcast_sub(&mean).unwrap();
                            let variance = x_centered.sqr().unwrap().mean(last_dim).unwrap();
                            let std = (variance + 1e-5).unwrap().sqrt().unwrap();
                            let std = std.unsqueeze(last_dim).unwrap();
                            let mut normed = x_centered.broadcast_div(&std).unwrap();
                            let ln1_w = if ln1_w.dtype() != normed.dtype() {
                                ln1_w.to_dtype(normed.dtype()).unwrap()
                            } else {
                                ln1_w.clone()
                            };
                            let ln1_b = if ln1_b.dtype() != normed.dtype() {
                                ln1_b.to_dtype(normed.dtype()).unwrap()
                            } else {
                                ln1_b.clone()
                            };
                            normed = normed.broadcast_mul(&ln1_w).unwrap().broadcast_add(&ln1_b).unwrap();

                            // Attention
                            let qkv = normed.broadcast_matmul(&qkv_weight.unsqueeze(0).unwrap()).unwrap();
                            let qkv = qkv.reshape((1, *seq_len, 3, *num_heads, *head_dim)).unwrap();
                            let qkv = qkv.permute((0, 3, 2, 1, 4)).unwrap();
                            let q = qkv.get_on_dim(2, 0).unwrap();
                            let v = qkv.get_on_dim(2, 1).unwrap();
                            let k = qkv.get_on_dim(2, 2).unwrap();

                            let scale = 1.0 / (*head_dim as f64).sqrt();
                            let scores = q.broadcast_matmul(&k.transpose(2, 3).unwrap()).unwrap();
                            let scores = (scores * scale).unwrap();

                            let weights = candle_nn::ops::softmax(&scores, 3).unwrap();
                            let context = weights.broadcast_matmul(&v).unwrap();
                            let context = context.permute((0, 2, 1, 3)).unwrap()
                                .reshape((1, *seq_len, *hidden_dim)).unwrap();
                            let attn_out = context.broadcast_matmul(&out_weight.unsqueeze(0).unwrap()).unwrap();

                            // Residual 1
                            x = x.broadcast_add(&attn_out).unwrap();

                            // LN2
                            let last_dim = x.dims().len() - 1;
                            let mean = x.mean(last_dim).unwrap();
                            let mean = mean.unsqueeze(last_dim).unwrap();
                            let x_centered = x.broadcast_sub(&mean).unwrap();
                            let variance = x_centered.sqr().unwrap().mean(last_dim).unwrap();
                            let std = (variance + 1e-5).unwrap().sqrt().unwrap();
                            let std = std.unsqueeze(last_dim).unwrap();
                            let mut normed = x_centered.broadcast_div(&std).unwrap();
                            let ln2_w = if ln2_w.dtype() != normed.dtype() {
                                ln2_w.to_dtype(normed.dtype()).unwrap()
                            } else {
                                ln2_w.clone()
                            };
                            let ln2_b = if ln2_b.dtype() != normed.dtype() {
                                ln2_b.to_dtype(normed.dtype()).unwrap()
                            } else {
                                ln2_b.clone()
                            };
                            normed = normed.broadcast_mul(&ln2_w).unwrap().broadcast_add(&ln2_b).unwrap();

                            // FFN
                            let hidden = normed.broadcast_matmul(&fc_in.unsqueeze(0).unwrap()).unwrap();
                            let activated = hidden.gelu().unwrap();
                            let ffn_out = activated.broadcast_matmul(&fc_out.unsqueeze(0).unwrap()).unwrap();

                            // Residual 2
                            x = x.broadcast_add(&ffn_out).unwrap();
                        }
                    });
                },
            );
        }
    }
    group.finish();
}

pub fn bench_e2e_inference(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("e2e_inference");

    for &hidden_dim in &[256, 512, 1024] {
        let seq_len = 7; // prefill
        let num_heads = 8;
        let head_dim = hidden_dim / num_heads;
        let ffn_dim = hidden_dim * 4;
        let num_layers = 6;

        let mut layers = Vec::new();
        for _ in 0..num_layers {
            let qkv_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), &device).unwrap();
            let out_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device).unwrap();
            let ln1_w = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
            let ln1_b = Tensor::zeros(hidden_dim, DType::F32, &device).unwrap();
            let ln2_w = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
            let ln2_b = Tensor::zeros(hidden_dim, DType::F32, &device).unwrap();
            let fc_in = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, ffn_dim), &device).unwrap();
            let fc_out = Tensor::randn(0.0f32, 0.02f32, (ffn_dim, hidden_dim), &device).unwrap();
            layers.push((qkv_weight, out_weight, ln1_w, ln1_b, ln2_w, ln2_b, fc_in, fc_out));
        }

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("prefill", format!("h{}", hidden_dim)),
            &(&layers, hidden_dim, num_layers, num_heads, head_dim, ffn_dim, seq_len),
            |b, (layers, hidden_dim, num_layers, num_heads, head_dim, ffn_dim, seq_len)| {
                b.iter(|| {
                    let mut x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap();
                    for layer in layers.iter().take(*num_layers) {
                        let (qkv_weight, out_weight, ln1_w, ln1_b, ln2_w, ln2_b, fc_in, fc_out) = layer;

                        let last_dim = x.dims().len() - 1;
                        let mean = x.mean(last_dim).unwrap();
                        let mean = mean.unsqueeze(last_dim).unwrap();
                        let x_centered = x.broadcast_sub(&mean).unwrap();
                        let variance = x_centered.sqr().unwrap().mean(last_dim).unwrap();
                        let std = (variance + 1e-5).unwrap().sqrt().unwrap();
                        let std = std.unsqueeze(last_dim).unwrap();
                        let mut normed = x_centered.broadcast_div(&std).unwrap();
                        let ln1_w = if ln1_w.dtype() != normed.dtype() {
                            ln1_w.to_dtype(normed.dtype()).unwrap()
                        } else {
                            ln1_w.clone()
                        };
                        let ln1_b = if ln1_b.dtype() != normed.dtype() {
                            ln1_b.to_dtype(normed.dtype()).unwrap()
                        } else {
                            ln1_b.clone()
                        };
                        normed = normed.broadcast_mul(&ln1_w).unwrap().broadcast_add(&ln1_b).unwrap();

                        let qkv = normed.broadcast_matmul(&qkv_weight.unsqueeze(0).unwrap()).unwrap();
                        let qkv = qkv.reshape((1, *seq_len, 3, *num_heads, *head_dim)).unwrap();
                        let qkv = qkv.permute((0, 3, 2, 1, 4)).unwrap();
                        let q = qkv.get_on_dim(2, 0).unwrap();
                        let v = qkv.get_on_dim(2, 1).unwrap();
                        let k = qkv.get_on_dim(2, 2).unwrap();

                        let scale = 1.0 / (*head_dim as f64).sqrt();
                        let scores = q.broadcast_matmul(&k.transpose(2, 3).unwrap()).unwrap();
                        let scores = (scores * scale).unwrap();

                        let weights = candle_nn::ops::softmax(&scores, 3).unwrap();
                        let context = weights.broadcast_matmul(&v).unwrap();
                        let context = context.permute((0, 2, 1, 3)).unwrap()
                            .reshape((1, *seq_len, *hidden_dim)).unwrap();
                        let attn_out = context.broadcast_matmul(&out_weight.unsqueeze(0).unwrap()).unwrap();

                        x = x.broadcast_add(&attn_out).unwrap();
                    }
                });
            },
        );
    }
    group.finish();
}

pub fn bench_f16_vs_f32(c: &mut Criterion) {
    let device = Device::Cpu;
    let mut group = c.benchmark_group("f16_vs_f32");

    for &dtype in &[DType::F32, DType::F16] {
        let hidden_dim = 1024;
        let num_heads = 16;
        let head_dim = hidden_dim / num_heads;
        let seq_len = 64;

        let qkv_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim * 3), &device).unwrap().to_dtype(dtype).unwrap();
        let out_weight = Tensor::randn(0.0f32, 0.02f32, (hidden_dim, hidden_dim), &device).unwrap().to_dtype(dtype).unwrap();

        group.throughput(Throughput::Elements((seq_len * hidden_dim * num_heads) as u64));
        group.bench_with_input(
            BenchmarkId::new("attention", format!("{:?}", dtype)),
            &(&qkv_weight, &out_weight, hidden_dim, num_heads, head_dim, seq_len),
            |b, (qkv_weight, out_weight, hidden_dim, num_heads, head_dim, seq_len)| {
                b.iter(|| {
                    let x = Tensor::randn(0.0f32, 1.0f32, (1, *seq_len, *hidden_dim), &device).unwrap().to_dtype(dtype).unwrap();
                    let qkv = x.broadcast_matmul(&qkv_weight.unsqueeze(0).unwrap()).unwrap();
                    let qkv = qkv.reshape((1, *seq_len, 3, *num_heads, *head_dim)).unwrap();
                    let qkv = qkv.permute((0, 3, 2, 1, 4)).unwrap();
                    let q = qkv.get_on_dim(2, 0).unwrap();
                    let v = qkv.get_on_dim(2, 1).unwrap();
                    let k = qkv.get_on_dim(2, 2).unwrap();

                    let scale = 1.0 / (*head_dim as f64).sqrt();
                    let scores = q.broadcast_matmul(&k.transpose(2, 3).unwrap()).unwrap();
                    let scores = (scores * scale).unwrap();

                    let weights = candle_nn::ops::softmax(&scores, 3).unwrap();
                    let context = weights.broadcast_matmul(&v).unwrap();
                    let context = context.permute((0, 2, 1, 3)).unwrap()
                        .reshape((1, *seq_len, *hidden_dim)).unwrap();
                    context.broadcast_matmul(&out_weight.unsqueeze(0).unwrap())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_attention, bench_ffn, bench_layernorm, bench_full_block, bench_e2e_inference, bench_f16_vs_f32);
criterion_main!(benches);