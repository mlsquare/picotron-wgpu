//! Context parallelism implementation

use anyhow::Result;

/// Context parallelism manager
pub struct ContextParallel {
    world_size: usize,
    rank: usize,
}

impl ContextParallel {
    /// Create new context parallel manager
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
