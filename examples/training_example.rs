//! PicoTron WGPU Training Example
//! 
//! This example demonstrates how to train a PicoTron model on a small corpus
//! using simple gradient descent and cross-entropy loss.

use picotron_wgpu::*;
use anyhow::Result;
use log::info;

/// Small training corpus
const TRAINING_CORPUS: &[&str] = &[
    "hello world",
    "rust is awesome",
    "machine learning",
    "webgpu rocks",
    "neural networks",
    "deep learning",
    "artificial intelligence",
    "computer science",
    "programming languages",
    "data structures",
    "algorithms and complexity",
    "software engineering",
    "distributed systems",
    "cloud computing",
    "cybersecurity",
    "blockchain technology",
    "quantum computing",
    "robotics and automation",
    "natural language processing",
    "computer vision",
];

/// Create training data from corpus
fn create_training_data(tokenizer: &SimpleTokenizer) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
    let mut training_data = Vec::new();
    
    for text in TRAINING_CORPUS {
        let tokens = tokenizer.encode(text);
        if tokens.len() > 1 {
            // Create input-target pairs for next token prediction
            for i in 0..tokens.len() - 1 {
                let input = tokens[..i+1].to_vec();
                // Target should be the same length as input, with the next token at the end
                let mut target = input.clone();
                if i + 1 < tokens.len() {
                    target[i] = tokens[i + 1]; // Replace last token with next token
                }
                training_data.push((input, target));
            }
        }
    }
    
    info!("Created {} training examples", training_data.len());
    Ok(training_data)
}

/// Train the model for one epoch
fn train_epoch(
    model: &mut PicoTronModel,
    training_data: &[(Vec<u32>, Vec<u32>)],
    learning_rate: f32,
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut batch_count = 0;
    
    for (input_ids, target_ids) in training_data {
        // Compute loss and gradients
        let loss = model.compute_loss_and_gradients(input_ids, target_ids)?;
        total_loss += loss;
        batch_count += 1;
        
        // Update parameters
        model.update_parameters(learning_rate);
    }
    
    let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
    Ok(avg_loss)
}

/// Evaluate the model on training data
fn evaluate_model(
    model: &PicoTronModel,
    training_data: &[(Vec<u32>, Vec<u32>)],
) -> Result<f32> {
    let mut total_loss = 0.0;
    let mut batch_count = 0;
    
    for (input_ids, target_ids) in training_data {
        let loss = model.evaluate_loss(input_ids, target_ids)?;
        total_loss += loss;
        batch_count += 1;
    }
    
    let avg_loss = if batch_count > 0 { total_loss / batch_count as f32 } else { 0.0 };
    Ok(avg_loss)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("🚀 PicoTron WGPU Training Example");
    println!("==================================");
    
    // Create a small model configuration for training
    let mut config = PicoTronConfig::default();
    config.model.vocab_size = 100;  // Small vocabulary for demo
    config.model.hidden_size = 64;  // Small hidden size for demo
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
    
    // Create training data
    println!("📚 Creating training data from corpus...");
    let training_data = create_training_data(&tokenizer)?;
    println!("✅ Created {} training examples", training_data.len());
    
    // Show some examples
    println!("\n📝 Training Examples:");
    for (i, (input, target)) in training_data.iter().take(5).enumerate() {
        let input_text = tokenizer.decode(input);
        let target_text = tokenizer.decode(target);
        println!("  {}. Input: \"{}\" -> Target: \"{}\"", i + 1, input_text, target_text);
    }
    
    // Create model with random weights
    println!("\n🧠 Creating model with random weights...");
    let mut model = PicoTronModel::new(config.model).await?;
    println!("✅ Model created successfully");
    
    // Get device info
    let device_info = model.device().get_info();
    println!("🖥️  GPU Device: {}", device_info.name);
    println!("🔧 Backend: {:?}", device_info.backend);
    
    // Training parameters
    let epochs = 10;
    let learning_rate = 0.01;
    
    println!("\n🎯 Starting Training:");
    println!("====================");
    println!("Epochs: {}", epochs);
    println!("Learning Rate: {}", learning_rate);
    println!("Training Examples: {}", training_data.len());
    
    // Initial evaluation
    let initial_loss = evaluate_model(&model, &training_data)?;
    println!("\n📊 Initial Loss: {:.4}", initial_loss);
    
    // Training loop
    for epoch in 0..epochs {
        // Train for one epoch
        let train_loss = train_epoch(&mut model, &training_data, learning_rate)?;
        
        // Evaluate on training data
        let eval_loss = evaluate_model(&model, &training_data)?;
        
        println!("Epoch {}/{}: Train Loss: {:.4}, Eval Loss: {:.4}", 
                 epoch + 1, epochs, train_loss, eval_loss);
    }
    
    // Test generation after training
    println!("\n🎲 Testing Generation After Training:");
    println!("=====================================");
    
    let test_prompts = vec![
        "hello",
        "rust",
        "machine",
        "webgpu",
        "neural",
    ];
    
    for (i, prompt) in test_prompts.iter().enumerate() {
        println!("\n📝 Prompt {}: \"{}\"", i + 1, prompt);
        
        // Tokenize the prompt
        let input_tokens = tokenizer.encode(prompt);
        println!("🔢 Input tokens: {:?}", input_tokens);
        
        // Generate text
        let max_length = 15;
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
    
    // Final evaluation
    let final_loss = evaluate_model(&model, &training_data)?;
    println!("\n📊 Final Loss: {:.4}", final_loss);
    println!("📈 Loss Improvement: {:.4}", initial_loss - final_loss);
    
    // Test forward pass
    println!("\n🧮 Testing Forward Pass:");
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
    
    println!("\n🎉 Training example completed successfully!");
    println!("💡 The model has been trained on a small corpus and should show");
    println!("   some improvement in loss and potentially better text generation.");
    
    Ok(())
}
