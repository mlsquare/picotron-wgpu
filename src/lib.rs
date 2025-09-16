//! PicoTron WGPU Implementation
//! 
//! A minimalistic distributed training framework for LLaMA-like models
//! using WebGPU for cross-platform GPU acceleration.

pub mod config;
pub mod model;
pub mod training;
pub mod parallelism;
pub mod gpu;
pub mod utils;
pub mod tokenizer;

pub use config::*;
pub use model::*;
pub use training::*;
pub use parallelism::*;
pub use gpu::*;
pub use utils::*;
pub use tokenizer::*;

use log::{info, warn, error};
use anyhow::Result;

/// Initialize PicoTron with logging
pub fn init() -> Result<()> {
    env_logger::init();
    info!("Initializing PicoTron WGPU");
    Ok(())
}

/// PicoTron version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get PicoTron version
pub fn version() -> &'static str {
    VERSION
}
