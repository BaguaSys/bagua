fn main() {
    let nvcc_path = which::which("nvcc").expect("cannot find nvcc");
    let cuda_home = nvcc_path
        .parent()
        .expect("cannot find nvcc parent directory")
        .parent()
        .expect("cannot find nvcc parent directory")
        .display();
    let supported_sms = cmd_lib::run_fun!(
        bash -c "nvcc --help | sed -n -e '/gpu-architecture <arch>/,/gpu-code <code>/ p' | sed -n -e '/Allowed values/,/gpu-code <code>/ p' | grep -i sm_ | grep -Eo 'sm_[0-9]+' | sed -e s/sm_//g | sort -g -u | tr '\n' ' '"
    ).unwrap();
    let supported_sms = supported_sms.strip_suffix(' ').unwrap().split(' ');
    let mut cuda_cc = cc::Build::new();
    cuda_cc
        .cuda(true)
        .opt_level(3)
        .include("cpp/include")
        .include("third_party/cub-1.8.0")
        .include("../../bagua/.data/include")
        .flag("-std=c++14")
        .flag("-cudart=shared");
    for sm in supported_sms {
        cuda_cc
            .flag("-gencode")
            .flag(format!("arch=compute_{},code=sm_{}", sm, sm).as_str());
    }
    cuda_cc
        .file("kernels/bagua_kernels.cu")
        .compile("libbagua_kernels.a");

    let mut cpp_builder = cpp_build::Config::new();
    cpp_builder.include(format!("{}/include", cuda_home));
    cpp_builder.include("cpp/include");
    cpp_builder.include("../../bagua/.data/include");
    cpp_builder.build("src/lib.rs");

    println!(
        "cargo:rustc-link-search=native={}",
        format!("{}/lib64", cuda_home)
    );

    println!("cargo:rustc-link-search=../bagua/.data/lib");
    println!("cargo:rustc-link-lib=nccl");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");
    shadow_rs::new().unwrap();
}
