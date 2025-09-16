// Matrix multiplication shader for PicoTron
// Implements efficient matrix multiplication for transformer operations

@group(0) @binding(0)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(1)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(2)
var<storage, read_write> matrix_c: array<f32>;

@group(0) @binding(3)
var<uniform> params: MatMulParams;

struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
}

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    // Bounds check
    if (row >= params.m || col >= params.n) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Compute dot product of row from A and column from B
    for (var i: u32 = 0u; i < params.k; i++) {
        let a_idx = row * params.k + i;
        let b_idx = i * params.n + col;
        sum += matrix_a[a_idx] * matrix_b[b_idx];
    }
    
    // Store result with alpha scaling
    let c_idx = row * params.n + col;
    matrix_c[c_idx] = params.alpha * sum + params.beta * matrix_c[c_idx];
}
