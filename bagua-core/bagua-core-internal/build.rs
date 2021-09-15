fn main() {
    let nvcc_path = which::which("nvcc")
        .expect("Cannot find nvcc, please install CUDA Toolkit and make sure nvcc is in your PATH first. See https://developer.nvidia.com/cuda-downloads");
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
        .include("cpp/include")
        .include("third_party/cub-1.8.0")
        .include("../python/bagua_core/.data/include")
        .flag("-std=c++14")
        .flag("-cudart=shared");

    if std::env::var("PROFILE").unwrap() == "release" {
        for sm in supported_sms {
            cuda_cc
                .flag("-gencode")
                .flag(format!("arch=compute_{},code=sm_{}", sm, sm).as_str());
        }
    }
    cuda_cc
        .file("kernels/bagua_kernels.cu")
        .compile("libbagua_kernels.a");

    let third_party_path = std::env::current_dir().unwrap();
    let bagua_data_path = std::env::current_dir().unwrap();
    let third_party_path = third_party_path.join("third_party");
    let bagua_data_path = bagua_data_path.join("../python/bagua_core/.data");
    let _al_builder = cmake::Config::new("third_party/Aluminum")
        .define("ALUMINUM_ENABLE_NCCL", "YES")
        .define("CUB_INCLUDE_PATH", third_party_path.join("cub-1.8.0"))
        .define("NCCL_LIBRARY", bagua_data_path.join("lib/libnccl.so"))
        .define("NCCL_INCLUDE_PATH", bagua_data_path.join("include"))
        .define("BUILD_SHARED_LIBS", "off")
        .out_dir(bagua_data_path.as_path().to_str().unwrap())
        .always_configure(true)
        .build();

    let mut cpp_builder = cpp_build::Config::new();
    cpp_builder.include(format!("{}/include", cuda_home));
    cpp_builder.include("cpp/include");
    let mpi_include_dirs = cmd_lib::run_fun!(bash -c "mpicxx --showme:incdirs").unwrap();
    let mpi_include_dirs: Vec<&str> = mpi_include_dirs.split(' ').collect();
    for mpi_include_dir in mpi_include_dirs.iter() {
        cpp_builder.include(mpi_include_dir);
    }
    cpp_builder.include(third_party_path.join("cub-1.8.0"));
    cpp_builder.include(bagua_data_path.join("include"));
    cpp_builder.build("src/lib.rs");

    let mpi_lib_dirs = cmd_lib::run_fun!(bash -c "mpicxx --showme:libdirs").unwrap();
    let mpi_lib_dirs: Vec<&str> = mpi_lib_dirs.split(' ').collect();
    for mpi_lib_dir in mpi_lib_dirs.iter() {
        println!("cargo:rustc-link-search={}", mpi_lib_dir);
    }
    println!(
        "cargo:rustc-link-search=native={}",
        format!("{}/lib64", cuda_home)
    );
    println!(
        "cargo:rustc-link-search={}",
        bagua_data_path.join("lib").as_path().to_str().unwrap()
    );
    println!(
        "cargo:rustc-link-search={}",
        bagua_data_path.join("lib64").as_path().to_str().unwrap()
    );
    println!("cargo:rustc-link-lib=static=Al");
    println!("cargo:rustc-link-lib=mpi");
    println!("cargo:rustc-link-lib=nccl");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=nvrtc");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-changed=src/");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=build.rs");

    // bindgen --allowlist-type '.*TensorImpl.*' --enable-cxx-namespaces --ignore-functions --ignore-methods --size_t-is-usize --default-enum-style=rust --opaque-type 'std.*' --opaque-type 'c10::optional.*' wrapper.h -- -x c++ -std=c++14 > src/torch_ffi.rs
    shadow_rs::new().unwrap();
}
