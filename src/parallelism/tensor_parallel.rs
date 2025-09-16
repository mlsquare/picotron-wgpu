//! Tensor parallelism implementation

use anyhow::Result;

/// Tensor parallelism manager
pub struct TensorParallel {
    world_size: usize,
    rank: usize,
}

impl TensorParallel {
    /// Create new tensor parallel manager
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self { world_size, rank }
    }
    
    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    
    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
    }
}
