//! Training loop for PicoTron

use crate::config::TrainingConfig;
use anyhow::Result;

/// PicoTron trainer
pub struct PicoTronTrainer {
    config: TrainingConfig,
}

impl PicoTronTrainer {
    /// Create a new trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }
    
    /// Get training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
}
