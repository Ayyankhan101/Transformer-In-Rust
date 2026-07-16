#![allow(dead_code)]

use crate::tokenizer::CodeGenTokenizer;
use candle_core::Result;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use std::io::Read;
use std::path::Path;

#[derive(Clone)]
pub struct TrainingExample {
    pub tokens: Vec<u32>,
}

pub struct DataLoader {
    examples: Vec<TrainingExample>,
    current_idx: usize,
    rng: StdRng,
    max_seq_len: usize,
}

impl DataLoader {
    pub fn new(
        data_dir: &Path,
        tokenizer: &CodeGenTokenizer,
        max_seq_len: usize,
        seed: u64,
    ) -> Result<Self> {
        let examples = load_data(data_dir, tokenizer)?;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut shuffled = examples;
        shuffled.shuffle(&mut rng);
        Ok(Self {
            examples: shuffled,
            current_idx: 0,
            rng,
            max_seq_len,
        })
    }

    pub fn from_examples(
        examples: Vec<TrainingExample>,
        max_seq_len: usize,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut shuffled = examples;
        shuffled.shuffle(&mut rng);
        Ok(Self {
            examples: shuffled,
            current_idx: 0,
            rng,
            max_seq_len,
        })
    }

    pub fn next_batch(&mut self, batch_size: usize) -> Vec<Vec<u32>> {
        let mut batch = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            if self.current_idx >= self.examples.len() {
                self.current_idx = 0;
                self.examples.shuffle(&mut self.rng);
            }

            let ex = &self.examples[self.current_idx];
            let tokens = if ex.tokens.len() > self.max_seq_len {
                let max_start = ex.tokens.len() - self.max_seq_len;
                let start = self.rng.gen_range(0..=max_start);
                ex.tokens[start..start + self.max_seq_len].to_vec()
            } else {
                ex.tokens.clone()
            };

            batch.push(tokens);
            self.current_idx += 1;
        }

        batch
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn reset(&mut self) {
        self.current_idx = 0;
        self.examples.shuffle(&mut self.rng);
    }
}

fn load_data(data_dir: &Path, tokenizer: &CodeGenTokenizer) -> Result<Vec<TrainingExample>> {
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
                        examples.push(TrainingExample { tokens: ids });
                    }
                }
            }
        }
    }

    Ok(examples)
}

pub fn download_default_data(dest_dir: &Path) -> Result<()> {
    const DEFAULT_DATA_URL: &str =
        "https://raw.githubusercontent.com/python/cpython/v3.12.0/Lib/functools.py";

    println!("  Downloading sample Python code from CPython repository...");
    let response = ureq::get(DEFAULT_DATA_URL)
        .call()
        .map_err(|e| candle_core::Error::Msg(format!("Download failed: {e}")))?;

    let mut body = Vec::new();
    let mut body_owned = response.into_body();
    let mut reader = body_owned.as_reader();
    reader
        .read_to_end(&mut body)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to read response: {e}")))?;

    std::fs::create_dir_all(dest_dir)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to create data dir: {e}")))?;

    let dest_path = dest_dir.join("functools.py");
    std::fs::write(&dest_path, &body)
        .map_err(|e| candle_core::Error::Msg(format!("Failed to write {:?}: {e}", dest_path)))?;

    println!("  Saved to {:?} ({} bytes)", dest_path, body.len());
    Ok(())
}

pub fn split_train_eval(
    examples: &[TrainingExample],
    train_split: f64,
) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
    let len = examples.len();
    if len <= 1 {
        return (examples.to_vec(), Vec::new());
    }
    let split_idx = (len as f64 * train_split) as usize;
    let split_idx = split_idx.max(1).min(len - 1);
    let train = examples[..split_idx].to_vec();
    let eval = examples[split_idx..].to_vec();
    (train, eval)
}
