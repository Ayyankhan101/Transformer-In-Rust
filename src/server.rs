#![allow(dead_code)]

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::codegen::config::CodeGenConfig;
use crate::codegen::weights::WeightLoader;
use crate::generation::codegen_generate::CodeGenGenerator;
use crate::tokenizer::CodeGenTokenizer;

pub struct AppState {
    generator: Arc<Mutex<CodeGenGenerator>>,
    tokenizer: CodeGenTokenizer,
}

#[derive(Deserialize)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_top_p")]
    pub top_p: f64,
}

fn default_max_tokens() -> usize { 128 }
fn default_temperature() -> f64 { 0.8 }
fn default_top_k() -> usize { 40 }
fn default_top_p() -> f64 { 0.9 }

#[derive(Serialize)]
pub struct GenerateResponse {
    pub generated: String,
    pub tokens: Vec<u32>,
    pub token_count: usize,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub model: String,
}

pub async fn start_server(
    weights_path: &str,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    let device = candle_core::Device::Cpu;
    let config = CodeGenConfig::default();

    println!("Loading CodeGen model from {weights_path}...");
    let model = WeightLoader::load_from_pytorch(
        std::path::Path::new(weights_path),
        &config,
        &device,
    )?;

    let tokenizer = CodeGenTokenizer::from_file("codegen_weights/tokenizer.json")
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

    let generator = CodeGenGenerator::new(
        model,
        0.8,  // temperature
        40,   // top_k
        0.9,  // top_p
        1.1,  // repetition_penalty
        256,  // max_new_tokens
    );

    let state = Arc::new(AppState {
        generator: Arc::new(Mutex::new(generator)),
        tokenizer,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate))
        .with_state(state);

    println!("Server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        model: "CodeGen-350M".to_string(),
    })
}

async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, (StatusCode, String)> {
    let prompt_ids = state.tokenizer.encode(&req.prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}")))?;

    let mut gen = state.generator.lock().await;
    gen.set_temperature(req.temperature);
    gen.set_top_k(req.top_k);
    gen.set_top_p(req.top_p);
    gen.set_max_new_tokens(req.max_tokens);

    let tokens = gen.generate(&prompt_ids)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Generation error: {e}")))?;

    let generated = state.tokenizer.decode(&tokens[prompt_ids.len()..])
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Decode error: {e}")))?;

    let token_count = tokens.len() - prompt_ids.len();

    Ok(Json(GenerateResponse {
        generated,
        tokens,
        token_count,
    }))
}
