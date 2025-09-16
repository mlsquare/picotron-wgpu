//! Model architecture for PicoTron

use crate::config::ModelConfig;
use crate::gpu::PicoTronDevice;
use anyhow::Result;
use rand::Rng;
use log::info;

/// PicoTron model implementation
pub struct PicoTronModel {
    config: ModelConfig,
    device: PicoTronDevice,
    // Simple random weights for demonstration
    embedding_weights: Vec<Vec<f32>>,
    attention_weights: Vec<Vec<f32>>,
    output_weights: Vec<Vec<f32>>,
    // Gradients for training
    embedding_gradients: Vec<Vec<f32>>,
    attention_gradients: Vec<Vec<f32>>,
    output_gradients: Vec<Vec<f32>>,
}

impl PicoTronModel {
    /// Create a new PicoTron model with random weights
    pub async fn new(config: ModelConfig) -> Result<Self> {
        info!("Creating PicoTron model with random weights");
        
        let device = PicoTronDevice::new().await?;
        
        // Initialize random weights for demonstration
        let mut rng = rand::thread_rng();
        
        // Embedding weights (vocab_size x hidden_size)
        let mut embedding_weights = Vec::new();
        for _ in 0..config.vocab_size {
            let mut row = Vec::new();
            for _ in 0..config.hidden_size {
                row.push(rng.gen_range(-0.1..0.1));
            }
            embedding_weights.push(row);
        }
        
        // Attention weights (hidden_size x hidden_size)
        let mut attention_weights = Vec::new();
        for _ in 0..config.hidden_size {
            let mut row = Vec::new();
            for _ in 0..config.hidden_size {
                row.push(rng.gen_range(-0.1..0.1));
            }
            attention_weights.push(row);
        }
        
        // Output weights (hidden_size x vocab_size)
        let mut output_weights = Vec::new();
        for _ in 0..config.hidden_size {
            let mut row = Vec::new();
            for _ in 0..config.vocab_size {
                row.push(rng.gen_range(-0.1..0.1));
            }
            output_weights.push(row);
        }
        
        info!("Model created with {} parameters", 
              config.vocab_size * config.hidden_size + 
              config.hidden_size * config.hidden_size + 
              config.hidden_size * config.vocab_size);
        
        // Initialize gradients
        let embedding_gradients = vec![vec![0.0; config.hidden_size]; config.vocab_size];
        let attention_gradients = vec![vec![0.0; config.hidden_size]; config.hidden_size];
        let output_gradients = vec![vec![0.0; config.vocab_size]; config.hidden_size];
        
        Ok(Self {
            config,
            device,
            embedding_weights,
            attention_weights,
            output_weights,
            embedding_gradients,
            attention_gradients,
            output_gradients,
        })
    }
    
    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    
    /// Get device
    pub fn device(&self) -> &PicoTronDevice {
        &self.device
    }
    
    /// Simple forward pass for inference
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
        }
        
        // Get embeddings for input tokens
        let mut hidden_states = Vec::new();
        for &token_id in input_ids {
            if token_id < self.embedding_weights.len() as u32 {
                hidden_states.extend_from_slice(&self.embedding_weights[token_id as usize]);
            } else {
                // Use zero embedding for unknown tokens
                hidden_states.resize(hidden_states.len() + self.config.hidden_size, 0.0);
            }
        }
        
        // Simple attention-like operation (just matrix multiplication for demo)
        let mut attended = vec![0.0; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            for j in 0..self.config.hidden_size {
                if i < hidden_states.len() && j < self.attention_weights.len() {
                    attended[i] += hidden_states[j] * self.attention_weights[j][i];
                }
            }
        }
        
        // Apply simple activation (ReLU-like)
        for val in &mut attended {
            *val = val.max(0.0);
        }
        
        // Output projection
        let mut logits = vec![0.0; self.config.vocab_size];
        for i in 0..self.config.vocab_size {
            for j in 0..self.config.hidden_size {
                if j < attended.len() && i < self.output_weights[j].len() {
                    logits[i] += attended[j] * self.output_weights[j][i];
                }
            }
        }
        
        // Apply softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        for logit in &mut logits {
            *logit = (*logit - max_logit).exp() / exp_sum;
        }
        
        Ok(logits)
    }
    
    /// Generate text using simple sampling
    pub fn generate(&self, prompt: &[u32], max_length: usize) -> Result<Vec<u32>> {
        let mut generated = prompt.to_vec();
        let mut rng = rand::thread_rng();
        
        info!("Generating text with prompt length: {}", prompt.len());
        
        while generated.len() < max_length {
            // Get the last few tokens for context (simple sliding window)
            let context_start = if generated.len() > 10 {
                generated.len() - 10
            } else {
                0
            };
            let context = &generated[context_start..];
            
            // Forward pass
            let logits = self.forward(context)?;
            
            // Sample from the distribution
            let mut cumulative = 0.0;
            let random_val: f32 = rng.gen();
            
            let mut next_token = 0;
            for (i, &prob) in logits.iter().enumerate() {
                cumulative += prob;
                if cumulative >= random_val {
                    next_token = i as u32;
                    break;
                }
            }
            
            generated.push(next_token);
            
            // Stop if we generate an end token (simple heuristic)
            if next_token == 0 || next_token >= self.config.vocab_size as u32 - 1 {
                break;
            }
        }
        
        info!("Generated {} tokens", generated.len() - prompt.len());
        Ok(generated)
    }
    
    /// Forward pass with gradient computation for training
    pub fn forward_with_gradients(&mut self, input_ids: &[u32]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Ok(vec![]);
        }
        
        // Get embeddings for input tokens
        let mut hidden_states = Vec::new();
        for &token_id in input_ids {
            if token_id < self.embedding_weights.len() as u32 {
                hidden_states.extend_from_slice(&self.embedding_weights[token_id as usize]);
            } else {
                // Use zero embedding for unknown tokens
                hidden_states.resize(hidden_states.len() + self.config.hidden_size, 0.0);
            }
        }
        
        // Simple attention-like operation (just matrix multiplication for demo)
        let mut attended = vec![0.0; self.config.hidden_size];
        for i in 0..self.config.hidden_size {
            for j in 0..self.config.hidden_size {
                if i < hidden_states.len() && j < self.attention_weights.len() {
                    attended[i] += hidden_states[j] * self.attention_weights[j][i];
                }
            }
        }
        
        // Apply simple activation (ReLU-like)
        for val in &mut attended {
            *val = val.max(0.0);
        }
        
        // Output projection
        let mut logits = vec![0.0; self.config.vocab_size];
        for i in 0..self.config.vocab_size {
            for j in 0..self.config.hidden_size {
                if j < attended.len() && i < self.output_weights[j].len() {
                    logits[i] += attended[j] * self.output_weights[j][i];
                }
            }
        }
        
        // Apply softmax
        let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        for logit in &mut logits {
            *logit = (*logit - max_logit).exp() / exp_sum;
        }
        
        Ok(logits)
    }
    
    /// Compute cross-entropy loss and gradients
    pub fn compute_loss_and_gradients(&mut self, input_ids: &[u32], target_ids: &[u32]) -> Result<f32> {
        if input_ids.len() != target_ids.len() {
            return Err(anyhow::anyhow!("Input and target lengths must match"));
        }
        
        // Forward pass
        let logits = self.forward_with_gradients(input_ids)?;
        
        // Compute cross-entropy loss
        let mut total_loss = 0.0;
        let mut loss_count = 0;
        
        for (i, &target_id) in target_ids.iter().enumerate() {
            if target_id < self.config.vocab_size as u32 {
                let target_idx = target_id as usize;
                if target_idx < logits.len() {
                    // Cross-entropy loss: -log(p_target)
                    let prob = logits[target_idx].max(1e-8); // Avoid log(0)
                    total_loss += -prob.ln();
                    loss_count += 1;
                    
                    // Gradient of cross-entropy loss w.r.t. logits
                    // For the target class: gradient = prob - 1
                    // For other classes: gradient = prob
                    for j in 0..logits.len() {
                        if j == target_idx {
                            // Target class gradient
                            let grad = logits[j] - 1.0;
                            // Backpropagate through output layer
                            for k in 0..self.config.hidden_size {
                                if k < self.output_gradients.len() && j < self.output_gradients[k].len() {
                                    self.output_gradients[k][j] += grad;
                                }
                            }
                        } else {
                            // Non-target class gradient
                            let grad = logits[j];
                            for k in 0..self.config.hidden_size {
                                if k < self.output_gradients.len() && j < self.output_gradients[k].len() {
                                    self.output_gradients[k][j] += grad;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let avg_loss = if loss_count > 0 { total_loss / loss_count as f32 } else { 0.0 };
        Ok(avg_loss)
    }
    
    /// Update model parameters using gradients (simple SGD)
    pub fn update_parameters(&mut self, learning_rate: f32) {
        // Update embedding weights
        for i in 0..self.embedding_weights.len() {
            for j in 0..self.embedding_weights[i].len() {
                self.embedding_weights[i][j] -= learning_rate * self.embedding_gradients[i][j];
                self.embedding_gradients[i][j] = 0.0; // Reset gradients
            }
        }
        
        // Update attention weights
        for i in 0..self.attention_weights.len() {
            for j in 0..self.attention_weights[i].len() {
                self.attention_weights[i][j] -= learning_rate * self.attention_gradients[i][j];
                self.attention_gradients[i][j] = 0.0; // Reset gradients
            }
        }
        
        // Update output weights
        for i in 0..self.output_weights.len() {
            for j in 0..self.output_weights[i].len() {
                self.output_weights[i][j] -= learning_rate * self.output_gradients[i][j];
                self.output_gradients[i][j] = 0.0; // Reset gradients
            }
        }
    }
    
    /// Get current loss on a dataset
    pub fn evaluate_loss(&self, input_ids: &[u32], target_ids: &[u32]) -> Result<f32> {
        if input_ids.len() != target_ids.len() {
            return Err(anyhow::anyhow!("Input and target lengths must match"));
        }
        
        // Forward pass (without gradients)
        let logits = self.forward(input_ids)?;
        
        // Compute cross-entropy loss
        let mut total_loss = 0.0;
        let mut loss_count = 0;
        
        for (i, &target_id) in target_ids.iter().enumerate() {
            if target_id < self.config.vocab_size as u32 {
                let target_idx = target_id as usize;
                if target_idx < logits.len() {
                    let prob = logits[target_idx].max(1e-8);
                    total_loss += -prob.ln();
                    loss_count += 1;
                }
            }
        }
        
        let avg_loss = if loss_count > 0 { total_loss / loss_count as f32 } else { 0.0 };
        Ok(avg_loss)
    }
}
