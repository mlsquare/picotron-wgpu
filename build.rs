use std::path::Path;

fn main() {
    // Re-run this script only if shader files change
    println!("cargo:rerun-if-changed=shaders/");
    
    // Compile all WGSL shaders to SPIR-V
    let shader_dir = Path::new("shaders");
    if shader_dir.exists() {
        for entry in std::fs::read_dir(shader_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("wgsl") {
                compile_shader(&path);
            }
        }
    }
}

fn compile_shader(shader_path: &Path) {
    println!("cargo:rerun-if-changed={}", shader_path.display());
    
    let shader_source = std::fs::read_to_string(shader_path)
        .expect(&format!("Failed to read shader file: {}", shader_path.display()));
    
    // Parse WGSL to Naga IR
    let module = naga::front::wgsl::parse_str(&shader_source)
        .expect(&format!("Failed to parse WGSL shader: {}", shader_path.display()));
    
    // Validate the shader
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .expect(&format!("Shader validation failed: {}", shader_path.display()));
    
    // Write the compiled shader to the output directory
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let shader_name = shader_path.file_stem().unwrap().to_str().unwrap();
    let shader_output_path = Path::new(&out_dir).join(format!("{}.spv", shader_name));
    
    // Convert to SPIR-V
    let spirv = naga::back::spv::write_vec(
        &module,
        &info,
        &naga::back::spv::Options::default(),
        None,
    ).expect(&format!("Failed to generate SPIR-V: {}", shader_path.display()));
    
    // Convert Vec<u32> to bytes
    let spirv_bytes: Vec<u8> = spirv.iter().flat_map(|&word| word.to_le_bytes()).collect();
    
    std::fs::write(&shader_output_path, spirv_bytes)
        .expect(&format!("Failed to write SPIR-V shader: {}", shader_path.display()));
}
