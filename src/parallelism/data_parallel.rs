//! Data parallelism implementation

use anyhow::Result;

/// Data parallelism manager
pub struct DataParallel {
    world_size: usize,
    rank: usize,
}

impl DataParallel {
    /// Create new data parallel manager
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
