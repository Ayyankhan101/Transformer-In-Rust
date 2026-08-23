//! HTTP inference server for CodeGen-350M.
//!
//! Built with axum, this module provides a REST API for code generation.
//! Requires the `server` feature flag.
//!
//! # Endpoints
//!
//! - `GET /health` — Health check
//! - `POST /generate` — Generate code from prompt

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use crate::generation::codegen_generate::CodeGenGenerator;
use crate::model::ModelContext;
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

fn default_max_tokens() -> usize {
    128
}
fn default_temperature() -> f64 {
    0.8
}
fn default_top_k() -> usize {
    40
}
fn default_top_p() -> f64 {
    0.9
}

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
    weights_dir: &Path,
    use_f16: bool,
    seed: Option<u64>,
    addr: SocketAddr,
) -> anyhow::Result<()> {
    println!("Loading CodeGen model from {}...", weights_dir.display());
    // Reuse the CLI loader so the server honours config.json and --f16 too.
    let mut ctx = ModelContext::load(weights_dir, use_f16, 0.8)?;
    ctx.generator.set_seed(seed);

    let state = Arc::new(AppState {
        generator: Arc::new(Mutex::new(ctx.generator)),
        tokenizer: ctx.tokenizer,
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
    let prompt_ids = state
        .tokenizer
        .encode(&req.prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Tokenization error: {e}")))?;

    let mut gen = state.generator.lock().await;
    gen.set_temperature(req.temperature);
    gen.set_top_k(req.top_k);
    gen.set_top_p(req.top_p);
    gen.set_max_new_tokens(req.max_tokens);

    let tokens = gen.generate(&prompt_ids).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Generation error: {e}"),
        )
    })?;

    // `generate` returns the generated tokens only — the prompt is not included.
    let generated = state.tokenizer.decode(&tokens).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Decode error: {e}"),
        )
    })?;

    let token_count = tokens.len();

    Ok(Json(GenerateResponse {
        generated,
        tokens,
        token_count,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_request_defaults() {
        let req: GenerateRequest = serde_json::from_str(r#"{"prompt": "fn main()"}"#).unwrap();
        assert_eq!(req.prompt, "fn main()");
        assert_eq!(req.max_tokens, 128);
        assert_eq!(req.temperature, 0.8);
        assert_eq!(req.top_k, 40);
        assert_eq!(req.top_p, 0.9);
    }

    #[test]
    fn test_generate_request_custom_values() {
        let req: GenerateRequest = serde_json::from_str(
            r#"{"prompt": "hello", "max_tokens": 256, "temperature": 0.5, "top_k": 10, "top_p": 0.8}"#,
        )
        .unwrap();
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.temperature, 0.5);
        assert_eq!(req.top_k, 10);
        assert_eq!(req.top_p, 0.8);
    }

    #[test]
    fn test_generate_response_serialization() {
        let resp = GenerateResponse {
            generated: "world".to_string(),
            tokens: vec![1, 2, 3],
            token_count: 3,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("world"));
        assert!(json.contains("token_count"));
    }

    #[test]
    fn test_health_response() {
        let resp = HealthResponse {
            status: "ok".to_string(),
            model: "CodeGen-350M".to_string(),
        };
        assert_eq!(resp.status, "ok");
        assert_eq!(resp.model, "CodeGen-350M");
    }
}
