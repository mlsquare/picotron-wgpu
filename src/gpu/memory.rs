//! GPU memory management for PicoTron

use wgpu::*;
use anyhow::Result;

/// GPU memory manager
pub struct GPUMemory {
    device: Device,
    queue: Queue,
}

impl GPUMemory {
    /// Create new GPU memory manager
    pub fn new(device: Device, queue: Queue) -> Self {
        Self { device, queue }
    }
    
    /// Create a buffer
    pub fn create_buffer(&self, size: u64, usage: BufferUsages) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    }
}
