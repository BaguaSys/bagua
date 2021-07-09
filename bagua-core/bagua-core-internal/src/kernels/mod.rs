use std::ffi::c_void;

#[link(name = "bagua_kernels", kind = "static")]
extern "C" {
    pub fn divide_inplace_f32_host(x: *mut c_void, D_: f32, N: i32, stream: *const c_void);
    pub fn divide_inplace_f16_host(x: *mut c_void, D_: f32, N: i32, stream: *const c_void);
    pub fn average_inplace_f32_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        stream: *const c_void,
    );
    pub fn average_inplace_f16_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        stream: *const c_void,
    );
    pub fn substract_inplace_f32_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        stream: *const c_void,
    );
    pub fn substract_inplace_f16_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        stream: *const c_void,
    );
    pub fn add_inplace_f32_host(x: *mut c_void, y: *const c_void, N: i32, stream: *const c_void);
    pub fn add_inplace_f16_host(x: *mut c_void, y: *const c_void, N: i32, stream: *const c_void);
    pub fn addmul_inplace_f32_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        factor: f32,
        stream: *const c_void,
    );
    pub fn addmul_inplace_f16_host(
        x: *mut c_void,
        y: *const c_void,
        N: i32,
        factor: f32,
        stream: *const c_void,
    );
    pub fn reduce_mean_f32_inplace_host(
        input: *mut c_void,
        chunk_size: i32,
        num_chunks: i32,
        target_chunk: i32,
        stream: *const c_void,
    );
    pub fn reduce_mean_f16_inplace_host(
        input: *mut c_void,
        chunk_size: i32,
        num_chunks: i32,
        target_chunk: i32,
        stream: *const c_void,
    );
    pub fn reduce_sum_f32_inplace_host(
        input: *mut c_void,
        chunk_size: i32,
        num_chunks: i32,
        target_chunk: i32,
        stream: *const c_void,
    );
    pub fn reduce_sum_f16_inplace_host(
        input: *mut c_void,
        chunk_size: i32,
        num_chunks: i32,
        target_chunk: i32,
        stream: *const c_void,
    );
    /// temp_buffer size is the same as decompressed tensor
    /// target_chunk = -1 means compressing all chunks
    pub fn compress_f32_to_uint8_host(
        input: *mut c_void,
        input_num_element: i32,
        chunk_size: i32,
        num_chunks: i32,
        output: *mut c_void,
        output_size_bytes: usize,
        temp_buffer: *mut c_void,
        temp_buffer_size_bytes: usize,
        target_chunk: i32,
        stream: *const c_void,
    );
    pub fn decompress_uint8_to_f32_host(
        input: *mut c_void,
        input_size_bytes: usize,
        chunk_size: i32,
        num_chunks: i32,
        output: *mut c_void,
        stream: *const c_void,
    );
    pub fn compress_f16_to_uint8_host(
        input: *mut c_void,
        input_num_element: i32,
        chunk_size: i32,
        num_chunks: i32,
        output: *mut c_void,
        output_size_bytes: usize,
        temp_buffer: *mut c_void,
        temp_buffer_size_bytes: usize,
        target_chunk: i32,
        stream: *const c_void,
    );
    pub fn decompress_uint8_to_f16_host(
        input: *mut c_void,
        input_size_bytes: usize,
        chunk_size: i32,
        num_chunks: i32,
        output: *mut c_void,
        stream: *const c_void,
    );
    pub fn array_min_max_size_f32_host(
        input: *mut c_void,
        input_num_element: i32,
        output: *mut c_void,
        stream: *const c_void,
    ) -> usize;
    pub fn array_min_max_size_f16_host(
        input: *mut c_void,
        input_num_element: i32,
        output: *mut c_void,
        stream: *const c_void,
    ) -> usize;
}
