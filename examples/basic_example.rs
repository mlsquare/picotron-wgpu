//! Basic PicoTron WGPU example

use picotron_wgpu::*;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize PicoTron
    init()?;
    
    println!("PicoTron WGPU Version: {}", version());
    
    // Create WGPU device
    let device = PicoTronDevice::new().await?;
    let info = device.get_info();
    
    println!("GPU Device: {}", info.name);
    println!("Backend: {:?}", info.backend);
    println!("Device Type: {:?}", info.device_type);
    
    // Create a simple configuration
    let config = PicoTronConfig::default();
    config.validate()?;
    
    println!("Configuration validated successfully");
    println!("Model: {}", config.model.name);
    println!("Hidden Size: {}", config.model.hidden_size);
    println!("Attention Heads: {}", config.model.num_attention_heads);
    println!("Hidden Layers: {}", config.model.num_hidden_layers);
    
    // Save configuration
    config.to_json("config.json")?;
    println!("Configuration saved to config.json");
    
    Ok(())
}
