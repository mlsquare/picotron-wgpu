//! PicoTron WGPU Inference Example
//! 
//! This example demonstrates how to use PicoTron for text generation
//! with a randomly initialized model that produces gibberish output.

use picotron_wgpu::*;
use anyhow::Result;
use log::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🚀 PicoTron WGPU Inference Example");
    println!("=====================================");
    
    // Create a small model configuration for demo
    let mut config = PicoTronConfig::default();
    config.model.vocab_size = 100;  // Small vocabulary for demo
    config.model.hidden_size = 128; // Small hidden size for demo
    config.model.num_attention_heads = 4;
    config.model.num_hidden_layers = 2;
    config.model.max_position_embeddings = 50;
    
    println!("📊 Model Configuration:");
    println!("  - Vocabulary Size: {}", config.model.vocab_size);
    println!("  - Hidden Size: {}", config.model.hidden_size);
    println!("  - Attention Heads: {}", config.model.num_attention_heads);
    println!("  - Hidden Layers: {}", config.model.num_hidden_layers);
    println!("  - Max Position: {}", config.model.max_position_embeddings);
    
    // Validate configuration
    config.validate()?;
    println!("✅ Configuration validated successfully");
    
    // Create tokenizer
    let tokenizer = SimpleTokenizer::new(config.model.vocab_size);
    println!("🔤 Tokenizer created with {} tokens", tokenizer.vocab_size());
    
    // Create model with random weights
    println!("🧠 Creating model with random weights...");
    let model = PicoTronModel::new(config.model).await?;
    println!("✅ Model created successfully");
    
    // Get device info
    let device_info = model.device().get_info();
    println!("🖥️  GPU Device: {}", device_info.name);
    println!("🔧 Backend: {:?}", device_info.backend);
    
    // Test prompts
    let prompts = vec![
        "Hello world",
        "The quick brown fox",
        "Rust is awesome",
        "Machine learning",
        "WebGPU rocks",
    ];
    
    println!("\n🎯 Generating text with random model:");
    println!("=====================================");
    
    for (i, prompt) in prompts.iter().enumerate() {
        println!("\n📝 Prompt {}: \"{}\"", i + 1, prompt);
        
        // Tokenize the prompt
        let input_tokens = tokenizer.encode(prompt);
        println!("🔢 Input tokens: {:?}", input_tokens);
        
        // Generate text
        let max_length = 20;
        let generated_tokens = model.generate(&input_tokens, max_length)?;
        println!("🎲 Generated tokens: {:?}", generated_tokens);
        
        // Decode back to text
        let generated_text = tokenizer.decode(&generated_tokens);
        println!("📖 Generated text: \"{}\"", generated_text);
        
        // Show some statistics
        let original_length = input_tokens.len();
        let generated_length = generated_tokens.len() - original_length;
        println!("📊 Generated {} new tokens", generated_length);
    }
    
    // Test forward pass
    println!("\n🧮 Testing forward pass:");
    println!("========================");
    
    let test_tokens = vec![1, 5, 10, 15, 20];
    let logits = model.forward(&test_tokens)?;
    
    println!("📊 Input tokens: {:?}", test_tokens);
    println!("🎯 Output logits shape: {}", logits.len());
    
    // Find top 5 predictions
    let mut indexed_logits: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    println!("🏆 Top 5 predictions:");
    for (i, (token_id, prob)) in indexed_logits.iter().take(5).enumerate() {
        let token_char = tokenizer.decode(&[*token_id as u32]);
        println!("  {}. Token {} (\"{}\"): {:.4}", i + 1, token_id, token_char, prob);
    }
    
    println!("\n🎉 Inference example completed successfully!");
    println!("💡 Note: This model uses random weights, so the output is gibberish.");
    println!("   In a real scenario, you would load pre-trained weights.");
    
    Ok(())
}
