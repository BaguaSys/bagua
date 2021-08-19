use dynamic_pool::DynamicPool;
use sized_object_pool::{DynamicReset, SizedAllocatable, SizedPool};

#[derive(Debug)]
pub struct CudaMemory {
    pub ptr: u64,
    pub num_bytes: usize,
}

impl CudaMemory {
    pub fn new(bytes: usize) -> Self {
        let ptr = unsafe {
            cpp::cpp!([bytes as "size_t"] -> u64 as "void *"
            {
                int *ptr = 0;
                CUDACHECK(cudaMalloc(&ptr, bytes));
                return ptr;
            })
        };
        Self {
            ptr,
            num_bytes: bytes,
        }
    }
}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        let ptr = self.ptr;
        unsafe {
            cpp::cpp!([ptr as "void *"]
            {
                CUDACHECK(cudaFree(ptr));
            })
        };
    }
}

impl SizedAllocatable for CudaMemory {
    fn new(size: usize) -> Self {
        Self::new(size)
    }

    fn size(&self) -> usize {
        self.num_bytes
    }
}

impl DynamicReset for CudaMemory {
    fn reset(&mut self) {}
}

pub static CUDA_DEVICE_MEMORY_POOL: once_cell::sync::Lazy<Vec<SizedPool<CudaMemory>>> =
    once_cell::sync::Lazy::new(|| {
        let mut pools = Vec::new();
        for _ in 0..64 {
            pools.push(SizedPool::<CudaMemory>::new(0, 40, 2048))
        }
        pools
    });

#[derive(Debug)]
pub struct CudaEvent {
    pub event: u64,
}

impl CudaEvent {
    pub fn new() -> Self {
        let event = unsafe {
            cpp::cpp!([] -> u64 as "cudaEvent_t"
            {
                cudaEvent_t event = 0;
                CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
                return event;
            })
        };
        Self { event }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let event = self.event;
        unsafe {
            cpp::cpp!([event as "cudaEvent_t"]
            {
                CUDACHECK(cudaEventDestroy(event));
            })
        };
    }
}

impl DynamicReset for CudaEvent {
    fn reset(&mut self) {}
}

pub static CUDA_EVENT_POOL: once_cell::sync::Lazy<DynamicPool<CudaEvent>> =
    once_cell::sync::Lazy::new(|| DynamicPool::new(0, 10, CudaEvent::new));
