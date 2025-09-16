// Layer normalization shader for PicoTron
// Implements layer normalization with optional bias

@group(0) @binding(0)
var<storage, read> input: array<f32>;

@group(0) @binding(1)
var<storage, read> gamma: array<f32>;

@group(0) @binding(2)
var<storage, read> beta: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<uniform> params: LayerNormParams;

struct LayerNormParams {
    batch_size: u32,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
}

@compute @workgroup_size(64)
fn layer_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.seq_len;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Calculate batch and sequence indices
    let batch_idx = idx / params.seq_len;
    let seq_idx = idx % params.seq_len;
    
    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        let input_idx = batch_idx * params.seq_len * params.hidden_size + 
                       seq_idx * params.hidden_size + i;
        mean += input[input_idx];
    }
    mean /= f32(params.hidden_size);
    
    // Compute variance
    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        let input_idx = batch_idx * params.seq_len * params.hidden_size + 
                       seq_idx * params.hidden_size + i;
        let diff = input[input_idx] - mean;
        variance += diff * diff;
    }
    variance /= f32(params.hidden_size);
    
    // Compute standard deviation
    let std_dev = sqrt(variance + params.eps);
    
    // Apply layer normalization
    for (var i: u32 = 0u; i < params.hidden_size; i++) {
        let input_idx = batch_idx * params.seq_len * params.hidden_size + 
                       seq_idx * params.hidden_size + i;
        let output_idx = input_idx;
        
        let normalized = (input[input_idx] - mean) / std_dev;
        output[output_idx] = gamma[i] * normalized + beta[i];
    }
}
