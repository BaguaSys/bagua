pub unsafe fn cuda_memcpy_device_to_host_sync(host_ptr: u64, device_ptr: u64, num_bytes: i32) {
    cpp::cpp!([host_ptr as "void*", device_ptr as "void*", num_bytes as "int"]
    {
        CUDACHECK(cudaMemcpy(host_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost));
    });
}
