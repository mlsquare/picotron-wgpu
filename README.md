# PicoTron WGPU

A minimalistic 4D-parallelism distributed training framework for LLaMA-like models, implemented using WGPU (WebGPU) for cross-platform GPU acceleration.

## 🚀 Features

- **Cross-platform GPU acceleration** using WGPU/WebGPU
- **4D Parallelism support**: Data, Tensor, Pipeline, and Context parallelism
- **WGSL shaders** compiled to SPIR-V for maximum compatibility
- **Complete training pipeline** with gradient computation and optimization
- **Text generation** with character-level tokenization
- **Educational focus** with comprehensive examples and documentation

## 🏗️ Architecture

### Core Components

- **Model**: PicoTron model with embedding, attention, and output layers
- **Training**: Complete training pipeline with cross-entropy loss and SGD optimizer
- **GPU Operations**: WGPU-based matrix multiplication, attention, and layer normalization
- **Parallelism**: Framework for 4D parallelism (Data, Tensor, Pipeline, Context)
- **Tokenizer**: Simple character-level tokenizer for text processing

### GPU Backends

- **Metal** (macOS/iOS)
- **Vulkan** (Linux/Windows/Android)
- **DirectX 12** (Windows)
- **OpenGL** (fallback)

## 📦 Installation

### Prerequisites

- Rust 1.70+
- WGPU-compatible GPU drivers
- For macOS: Metal support
- For Linux: Vulkan drivers
- For Windows: DirectX 12 or Vulkan drivers

### Build

```bash
git clone https://github.com/mlsquare/picotron-wgpu.git
cd picotron-wgpu
cargo build --release
```

## 🎯 Quick Start

### Basic Example

```bash
cargo run --example basic_example
```

### Inference Example

```bash
cargo run --example inference_example
```

### Training Example

```bash
cargo run --example training_example
```

## 📚 Examples

### 1. Basic Initialization

```rust
use picotron_wgpu::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize WGPU device
    let device = PicoTronWgpuDevice::new().await?;
    
    // Create model configuration
    let config = PicoTronConfig::default();
    
    // Create model
    let model = PicoTronModel::new(config.model).await?;
    
    Ok(())
}
```

### 2. Text Generation

```rust
use picotron_wgpu::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = PicoTronConfig::default();
    let tokenizer = SimpleTokenizer::new(config.model.vocab_size);
    let model = PicoTronModel::new(config.model).await?;
    
    // Generate text
    let prompt = "hello";
    let input_tokens = tokenizer.encode(prompt);
    let generated_tokens = model.generate(&input_tokens, 20)?;
    let generated_text = tokenizer.decode(&generated_tokens);
    
    println!("Generated: {}", generated_text);
    Ok(())
}
```

### 3. Training

```rust
use picotron_wgpu::*;

#[tokio::main]
async fn main() -> Result<()> {
    let config = PicoTronConfig::default();
    let mut model = PicoTronModel::new(config.model).await?;
    
    // Training loop
    for epoch in 0..10 {
        let loss = model.compute_loss_and_gradients(&input_ids, &target_ids)?;
        model.update_parameters(0.01);
        println!("Epoch {}: Loss = {:.4}", epoch, loss);
    }
    
    Ok(())
}
```

## 🔧 Configuration

### Model Configuration

```rust
let mut config = PicoTronConfig::default();
config.model.vocab_size = 100;
config.model.hidden_size = 128;
config.model.num_attention_heads = 4;
config.model.num_hidden_layers = 2;
config.model.max_position_embeddings = 50;
```

### Training Parameters

- **Learning Rate**: 0.01 (default)
- **Batch Size**: Configurable
- **Epochs**: 10 (example)
- **Optimizer**: Simple SGD

## 🎮 GPU Operations

### Matrix Multiplication

```rust
let result = gpu_ops.matmul(&buffer_a, &buffer_b, &buffer_c, m, n, k).await?;
```

### Attention

```rust
let result = gpu_ops.attention(&query, &key, &value, &output, 
                              batch_size, num_heads, seq_len, head_dim).await?;
```

### Layer Normalization

```rust
let result = gpu_ops.layer_norm(&input, &output, &gamma, &beta,
                                batch_size, seq_len, hidden_size, eps).await?;
```

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with logging
RUST_LOG=debug cargo test
```

## 📊 Performance

### Benchmarks

- **Apple M3**: ~100ms for 1000 tokens generation
- **Training**: ~50ms per epoch on small corpus
- **Memory**: ~64MB for small model (100 vocab, 128 hidden)

### Optimization Tips

1. Use release builds: `cargo build --release`
2. Enable GPU-specific optimizations
3. Batch operations when possible
4. Use appropriate workgroup sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the original PicoTron concept
- **WGPU team** for the excellent cross-platform GPU abstraction
- **Rust community** for the amazing ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mlsquare/picotron-wgpu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mlsquare/picotron-wgpu/discussions)
- **Documentation**: [GitHub Wiki](https://github.com/mlsquare/picotron-wgpu/wiki)

---

**PicoTron WGPU** - Cross-platform GPU-accelerated neural network training in Rust 🦀⚡