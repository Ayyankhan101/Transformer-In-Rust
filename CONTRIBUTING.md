# Contributing to Transformer-In-Rust

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Ayyankhan101/Transformer-In-Rust.git
cd Transformer-In-Rust

# Build
cargo build

# Run tests
cargo test

# Run tests with all features
cargo test --features server

# Lint
cargo clippy -- -D warnings

# Format
cargo fmt --check
```

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- No clippy warnings (`cargo clippy -- -D warnings`)
- Keep `#![allow(dead_code)]` annotations scoped to individual items, not blanket module-level
- Document public items with rustdoc comments (`///` or `//!`)

## Testing

- All new features must include unit tests
- Tests live in `#[cfg(test)] mod tests` blocks at the bottom of each module
- Integration tests go in `tests/`
- Run the full suite before submitting: `cargo test --features server`

## Pull Requests

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Ensure `cargo fmt`, `cargo clippy`, and `cargo test` all pass
4. Update documentation if you changed public APIs
5. Open a PR against `master`

## Reporting Issues

Open an issue on GitHub with:
- What you expected
- What actually happened
- Steps to reproduce
- Your OS and Rust version (`rustc --version`)
