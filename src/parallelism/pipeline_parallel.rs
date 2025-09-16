//! Pipeline parallelism implementation

use anyhow::Result;

/// Pipeline parallelism manager
pub struct PipelineParallel {
    world_size: usize,
    rank: usize,
}

impl PipelineParallel {
    /// Create new pipeline parallel manager
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
