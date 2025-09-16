//! WGPU device management for PicoTron

use wgpu::*;
use log::{info, warn, error};
use anyhow::Result;

/// WGPU device wrapper for PicoTron
pub struct PicoTronDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub info: AdapterInfo,
}

impl PicoTronDevice {
    /// Create a new PicoTron device
    pub async fn new() -> Result<Self> {
        info!("Initializing PicoTron WGPU device...");
        
        // Create WGPU instance
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        // Request adapter
        let adapter = instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }).await
        .ok_or_else(|| anyhow::anyhow!("No suitable GPU adapter found"))?;
        
        let info = adapter.get_info();
        info!("Selected adapter: {:?}", info);
        
        // Create device and queue
        let (device, queue) = adapter.request_device(
            &DeviceDescriptor {
                label: Some("PicoTron Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
            },
            None,
        ).await
        .map_err(|e| anyhow::anyhow!("Failed to create device: {:?}", e))?;
        
        info!("PicoTron WGPU device initialized successfully");
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
        })
    }
    
    /// Get device information
    pub fn get_info(&self) -> &AdapterInfo {
        &self.info
    }
    
    /// Check if device supports required features
    pub fn supports_features(&self, features: Features) -> bool {
        self.device.features().contains(features)
    }
    
    /// Get device limits
    pub fn get_limits(&self) -> Limits {
        self.device.limits()
    }
    
    /// Create a command encoder
    pub fn create_command_encoder(&self, label: Option<&str>) -> CommandEncoder {
        self.device.create_command_encoder(&CommandEncoderDescriptor {
            label,
        })
    }
    
    /// Submit command buffer
    pub fn submit(&self, command_buffer: CommandBuffer) {
        self.queue.submit(std::iter::once(command_buffer));
    }
    
    /// Wait for device to be idle
    pub fn wait_idle(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

impl Drop for PicoTronDevice {
    fn drop(&mut self) {
        info!("Dropping PicoTron WGPU device");
    }
}
