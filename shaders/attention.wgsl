// Multi-head attention shader for PicoTron
// Implements scaled dot-product attention mechanism

@group(0) @binding(0)
var<storage, read> query: array<f32>;

@group(0) @binding(1)
var<storage, read> key: array<f32>;

@group(0) @binding(2)
var<storage, read> value: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@group(0) @binding(4)
var<storage, read_write> attention_weights: array<f32>;

@group(0) @binding(5)
var<uniform> params: AttentionParams;

struct AttentionParams {
    batch_size: u32,
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    scale: f32,
}

@compute @workgroup_size(64)
fn attention(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.num_heads * params.seq_len * params.seq_len;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Calculate indices
    let batch_idx = idx / (params.num_heads * params.seq_len * params.seq_len);
    let head_idx = (idx % (params.num_heads * params.seq_len * params.seq_len)) / (params.seq_len * params.seq_len);
    let seq_i = (idx % (params.seq_len * params.seq_len)) / params.seq_len;
    let seq_j = idx % params.seq_len;
    
    // Compute attention score
    var score: f32 = 0.0;
    for (var k: u32 = 0u; k < params.head_dim; k++) {
        let q_idx = batch_idx * params.num_heads * params.seq_len * params.head_dim + 
                   head_idx * params.seq_len * params.head_dim + 
                   seq_i * params.head_dim + k;
        let k_idx = batch_idx * params.num_heads * params.seq_len * params.head_dim + 
                   head_idx * params.seq_len * params.head_dim + 
                   seq_j * params.head_dim + k;
        
        score += query[q_idx] * key[k_idx];
    }
    
    // Apply scaling
    score *= params.scale;
    
    // Store attention weight
    attention_weights[idx] = score;
}

@compute @workgroup_size(64)
fn attention_softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_heads = params.batch_size * params.num_heads * params.seq_len;
    
    if (idx >= total_heads) {
        return;
    }
    
    // Calculate batch and head indices
    let batch_idx = idx / (params.num_heads * params.seq_len);
    let head_idx = (idx % (params.num_heads * params.seq_len)) / params.seq_len;
    let seq_i = idx % params.seq_len;
    
    // Find maximum for numerical stability
    var max_val: f32 = -3.4028235e+38f;
    for (var j: u32 = 0u; j < params.seq_len; j++) {
        let weight_idx = batch_idx * params.num_heads * params.seq_len * params.seq_len + 
                        head_idx * params.seq_len * params.seq_len + 
                        seq_i * params.seq_len + j;
        max_val = max(max_val, attention_weights[weight_idx]);
    }
    
    // Compute softmax
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < params.seq_len; j++) {
        let weight_idx = batch_idx * params.num_heads * params.seq_len * params.seq_len + 
                        head_idx * params.seq_len * params.seq_len + 
                        seq_i * params.seq_len + j;
        let exp_val = exp(attention_weights[weight_idx] - max_val);
        attention_weights[weight_idx] = exp_val;
        sum += exp_val;
    }
    
    // Normalize
    for (var j: u32 = 0u; j < params.seq_len; j++) {
        let weight_idx = batch_idx * params.num_heads * params.seq_len * params.seq_len + 
                        head_idx * params.seq_len * params.seq_len + 
                        seq_i * params.seq_len + j;
        attention_weights[weight_idx] /= sum;
    }
}

@compute @workgroup_size(64)
fn attention_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.num_heads * params.seq_len * params.head_dim;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Calculate indices
    let batch_idx = idx / (params.num_heads * params.seq_len * params.head_dim);
    let head_idx = (idx % (params.num_heads * params.seq_len * params.head_dim)) / (params.seq_len * params.head_dim);
    let seq_i = (idx % (params.seq_len * params.head_dim)) / params.head_dim;
    let head_k = idx % params.head_dim;
    
    // Compute weighted sum
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < params.seq_len; j++) {
        let weight_idx = batch_idx * params.num_heads * params.seq_len * params.seq_len + 
                        head_idx * params.seq_len * params.seq_len + 
                        seq_i * params.seq_len + j;
        let value_idx = batch_idx * params.num_heads * params.seq_len * params.head_dim + 
                       head_idx * params.seq_len * params.head_dim + 
                       j * params.head_dim + head_k;
        
        sum += attention_weights[weight_idx] * value[value_idx];
    }
    
    output[idx] = sum;
}
