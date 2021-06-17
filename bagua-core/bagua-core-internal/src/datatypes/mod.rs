use crate::comm_ops::centralized_full_precision_synchronous::CentralizedFullPrecisionSynchronous;
use crate::comm_ops::centralized_low_precision_synchronous::CentralizedLowPrecisionSynchronous;
use crate::comm_ops::decentralized_full_precision_synchronous::{
    DecentralizedFullPrecisionSynchronous, PeerSelectionMode,
};
use crate::comm_ops::python_ffi_op::PythonFFIOp;
use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaSingleCommunicator};
use crate::resource_pool::{CudaMemory, CUDA_DEVICE_MEMORY_POOL};
use crate::telemetry::TELEMETRY;
use crate::{kernels, BaguaCoreError};
use itertools::Itertools;
use parking_lot::{Mutex, RwLock};
use sized_object_pool::DynamicPoolItem;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BaguaTensorDtype {
    F32,
    F16,
    U8,
    I64,
}

impl BaguaTensorDtype {
    pub fn bytes(&self) -> usize {
        match self {
            BaguaTensorDtype::F32 => 4,
            BaguaTensorDtype::F16 => 2,
            BaguaTensorDtype::U8 => 1,
            BaguaTensorDtype::I64 => 8,
        }
    }

    pub fn to_nccl_datatype(&self) -> i32 {
        match self {
            BaguaTensorDtype::F32 => 7,
            BaguaTensorDtype::F16 => 6,
            BaguaTensorDtype::U8 => 1,
            BaguaTensorDtype::I64 => 4,
        }
    }
}

#[derive(Debug)]
pub struct BaguaTensorRaw {
    pub ptr: u64,
    pub num_elem_allocated: usize,
    pub dtype: BaguaTensorDtype,
    pub num_elem: usize,
    pub device_id: usize,
    pub pool_allocation: Option<DynamicPoolItem<CudaMemory>>,
}

#[derive(Clone, Debug)]
pub struct MinMaxUInt8CompressionParameters {}

impl MinMaxUInt8CompressionParameters {
    pub fn get_compressed_buffer_size(
        n_chunks: usize,
        chunk_size: usize,
        from_dtype: &BaguaTensorDtype,
    ) -> usize {
        #[inline]
        fn align_size(size: usize, align: usize) -> usize {
            ((size) + (align) - 1) / (align) * (align)
        }

        match from_dtype {
            BaguaTensorDtype::F32 => {
                let align_bytes = 32;
                // Note chunk_size is already aligned outside
                let compressed_align_bytes = align_size(chunk_size * n_chunks, align_bytes);
                let min_max_align_bytes = align_size(4 * 2, align_bytes) * n_chunks; // assume min max are both f32
                return compressed_align_bytes + min_max_align_bytes;
            }
            BaguaTensorDtype::F16 => {
                let align_bytes = 32;
                // Note chunk_size is already aligned outside
                let compressed_align_bytes = align_size(chunk_size * n_chunks, align_bytes);
                let min_max_align_bytes = align_size(2 * 2, align_bytes) * n_chunks; // assume min max are both f16
                return compressed_align_bytes + min_max_align_bytes;
            }
            BaguaTensorDtype::U8 => {
                unimplemented!()
            }
            BaguaTensorDtype::I64 => {
                unimplemented!()
            }
        }
    }

    pub fn get_temp_buffer_size(
        input_ptr: u64,
        input_size: usize,
        output_ptr: u64,
        stream_ptr: u64,
        from_dtype: &BaguaTensorDtype,
    ) -> usize {
        match from_dtype {
            BaguaTensorDtype::F32 => unsafe {
                kernels::array_min_max_size_f32_host(
                    input_ptr as _,
                    input_size as _,
                    output_ptr as _,
                    stream_ptr as _,
                )
            },

            BaguaTensorDtype::F16 => unsafe {
                kernels::array_min_max_size_f16_host(
                    input_ptr as _,
                    input_size as _,
                    output_ptr as _,
                    stream_ptr as _,
                )
            },
            BaguaTensorDtype::U8 => {
                unimplemented!()
            }
            BaguaTensorDtype::I64 => {
                unimplemented!()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorCompressionMethod {
    MinMaxUInt8(MinMaxUInt8CompressionParameters),
}

impl BaguaTensorRaw {
    pub fn divide_inplace(&mut self, stream_ptr: u64, divide_factor: f32) {
        let tensor_ptr = self.ptr;
        let total_num_elem = self.num_elem;
        unsafe {
            match self.dtype {
                BaguaTensorDtype::F32 => {
                    kernels::divide_inplace_f32_host(
                        tensor_ptr as _,
                        divide_factor as f32,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::divide_inplace_f16_host(
                        tensor_ptr as _,
                        divide_factor as f32,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::U8 => {
                    unimplemented!()
                }
                BaguaTensorDtype::I64 => {
                    unimplemented!()
                }
            }
        }
    }

    pub fn clone_from(&mut self, other: &Self, stream_ptr: u64) {
        assert_eq!(self.dtype, other.dtype);
        assert_eq!(self.num_elem, other.num_elem);
        unsafe {
            let src = other.ptr;
            let dst = self.ptr;
            let count = self.num_elem * self.dtype.bytes();
            cpp::cpp!([stream_ptr as "cudaStream_t", dst as "void *", src as "const void *", count as "size_t"]
            {
            CUDACHECK(cudaMemcpyAsync( dst, src , count, cudaMemcpyDeviceToDevice, stream_ptr));
            });
        }
    }

    pub fn average_inplace(&mut self, other: &Self, stream_ptr: u64) {
        assert_eq!(self.dtype, other.dtype);
        assert_eq!(self.num_elem, other.num_elem);
        let tensor_ptr = self.ptr;
        let total_num_elem = self.num_elem;
        unsafe {
            match self.dtype {
                BaguaTensorDtype::F32 => {
                    kernels::average_inplace_f32_host(
                        tensor_ptr as _,
                        other.ptr as _,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::average_inplace_f16_host(
                        tensor_ptr as _,
                        other.ptr as _,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::U8 => {
                    unimplemented!()
                }
                BaguaTensorDtype::I64 => {
                    unimplemented!()
                }
            }
        }
    }

    pub fn compress(
        &self,
        compression: &TensorCompressionMethod,
        n_chunks: usize,
        stream_ptr: u64,
        target_chunk: i32,
    ) -> Result<BaguaTensorRaw, BaguaCoreError> {
        match compression {
            TensorCompressionMethod::MinMaxUInt8(_parameters) => {
                assert_eq!(
                    self.num_elem_allocated % n_chunks,
                    0,
                    "compression tensor size % n_chunks must be 0"
                );
                let chunk_size = self.num_elem_allocated / n_chunks;
                let output_buffer_size =
                    MinMaxUInt8CompressionParameters::get_compressed_buffer_size(
                        n_chunks,
                        chunk_size,
                        &self.dtype,
                    );
                let output_buffer =
                    CUDA_DEVICE_MEMORY_POOL[self.device_id].try_pull(output_buffer_size)?;

                let temp_buffer_size = MinMaxUInt8CompressionParameters::get_temp_buffer_size(
                    self.ptr,
                    self.num_elem,
                    output_buffer.ptr,
                    stream_ptr,
                    &self.dtype,
                );
                let temp_buffer =
                    CUDA_DEVICE_MEMORY_POOL[self.device_id].try_pull(temp_buffer_size)?;

                match self.dtype {
                    BaguaTensorDtype::F32 => unsafe {
                        kernels::compress_f32_to_uint8_host(
                            self.ptr as _,
                            self.num_elem as _,
                            chunk_size as _,
                            n_chunks as _,
                            output_buffer.ptr as _,
                            output_buffer_size,
                            temp_buffer.ptr as _,
                            temp_buffer_size,
                            target_chunk,
                            stream_ptr as _,
                        );
                    },
                    BaguaTensorDtype::F16 => unsafe {
                        kernels::compress_f16_to_uint8_host(
                            self.ptr as _,
                            self.num_elem as _,
                            chunk_size as _,
                            n_chunks as _,
                            output_buffer.ptr as _,
                            output_buffer_size,
                            temp_buffer.ptr as _,
                            temp_buffer_size,
                            target_chunk,
                            stream_ptr as _,
                        );
                    },
                    BaguaTensorDtype::U8 => {
                        unimplemented!()
                    }
                    BaguaTensorDtype::I64 => {
                        unimplemented!()
                    }
                }
                return Ok(BaguaTensorRaw {
                    ptr: output_buffer.ptr,
                    num_elem_allocated: output_buffer_size,
                    dtype: BaguaTensorDtype::U8,
                    num_elem: output_buffer_size,
                    device_id: self.device_id,
                    pool_allocation: Some(output_buffer),
                });
            }
        }
    }

    pub fn decompress_from(
        &mut self,
        compression: &TensorCompressionMethod,
        n_chunks: usize,
        compressed_buffer: &BaguaTensorRaw,
        stream_ptr: u64,
    ) {
        assert_eq!(
            self.num_elem_allocated % n_chunks,
            0,
            "compression tensor size % n_chunks must be 0"
        );
        let chunk_size = self.num_elem_allocated / n_chunks;
        match compression {
            TensorCompressionMethod::MinMaxUInt8(_parameters) => match self.dtype {
                BaguaTensorDtype::F32 => unsafe {
                    kernels::decompress_uint8_to_f32_host(
                        compressed_buffer.ptr as _,
                        compressed_buffer.num_elem_allocated * compressed_buffer.dtype.bytes(),
                        chunk_size as _,
                        n_chunks as _,
                        self.ptr as _,
                        stream_ptr as _,
                    );
                },
                BaguaTensorDtype::F16 => unsafe {
                    kernels::decompress_uint8_to_f16_host(
                        compressed_buffer.ptr as _,
                        compressed_buffer.num_elem_allocated * compressed_buffer.dtype.bytes(),
                        chunk_size as _,
                        n_chunks as _,
                        self.ptr as _,
                        stream_ptr as _,
                    );
                },
                BaguaTensorDtype::U8 => {
                    unimplemented!()
                }
                BaguaTensorDtype::I64 => {
                    unimplemented!()
                }
            },
        }
    }

    pub fn reduce_mean_inplace(&mut self, n_chunks: usize, target_chunk: usize, stream_ptr: u64) {
        assert_eq!(
            self.num_elem_allocated % n_chunks,
            0,
            "reduce_mean_inplace requires tensor aligned"
        );
        let chunk_size = self.num_elem_allocated / n_chunks;
        match self.dtype {
            BaguaTensorDtype::F32 => unsafe {
                kernels::reduce_mean_f32_inplace_host(
                    self.ptr as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::F16 => unsafe {
                kernels::reduce_mean_f16_inplace_host(
                    self.ptr as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::U8 => {
                unimplemented!()
            }
            BaguaTensorDtype::I64 => {
                unimplemented!()
            }
        }
    }

    pub fn reduce_sum_inplace(&mut self, n_chunks: usize, target_chunk: usize, stream_ptr: u64) {
        assert_eq!(
            self.num_elem_allocated % n_chunks,
            0,
            "reduce_sum_inplace requires tensor aligned"
        );
        let chunk_size = self.num_elem_allocated / n_chunks;
        match self.dtype {
            BaguaTensorDtype::F32 => unsafe {
                kernels::reduce_sum_f32_inplace_host(
                    self.ptr as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::F16 => unsafe {
                kernels::reduce_sum_f16_inplace_host(
                    self.ptr as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::U8 => {
                unimplemented!()
            }
            BaguaTensorDtype::I64 => {
                unimplemented!()
            }
        }
    }
}

#[derive(Debug)]
pub struct BaguaTensorInner {
    pub raw: BaguaTensorRaw,
    pub ready_for_comm: bool,
    pub ready_cuda_event_ptr: u64,
}

#[derive(Clone, Debug)]
pub struct BaguaTensor {
    pub id: u64,
    pub inner: Arc<RwLock<BaguaTensorInner>>,
}

impl BaguaTensor {
    pub fn mark_comm_ready(&self, cuda_event_ptr: u64) {
        if cuda_event_ptr == 0 {
            tracing::info!("mark comm ready with an event 0, ignoring event");
        }
        match TELEMETRY.as_ref() {
            None => {}
            Some(ref x) => {
                x.lock().new_tensor_ready(self.id);
            }
        }
        let mut guard = self.inner.write();
        guard.ready_for_comm = true;
        guard.ready_cuda_event_ptr = cuda_event_ptr;
    }

    pub fn mark_comm_not_ready(&self) {
        self.inner.write().ready_for_comm = false;
    }

    pub fn ready_for_comm(&self) -> bool {
        self.inner.read().ready_for_comm
    }
}

impl BaguaTensor {
    pub fn new(
        ptr: u64,
        num_elem: usize,
        num_elem_allocated: usize,
        dtype: &str,
        device_id: usize,
    ) -> Self {
        let dtype = match dtype {
            "f32" => BaguaTensorDtype::F32,
            "f16" => BaguaTensorDtype::F16,
            "u8" => BaguaTensorDtype::U8,
            "i64" => BaguaTensorDtype::I64,
            _ => {
                unimplemented!()
            }
        };
        let id = lazy_id::Id::lazy().get();
        tracing::debug!("generate tensor id {}", id);
        Self {
            id,
            inner: Arc::new(RwLock::new(BaguaTensorInner {
                raw: BaguaTensorRaw {
                    ptr,
                    num_elem,
                    num_elem_allocated,
                    dtype,
                    device_id,
                    pool_allocation: None,
                },
                ready_for_comm: false,
                ready_cuda_event_ptr: 0,
            })),
        }
    }

    pub fn compress(&self, method: &str, n_chunks: usize, target_chunk: i32) -> Self {
        match method {
            "min_max_uint8" => Self {
                id: 0,
                inner: Arc::new(RwLock::new(BaguaTensorInner {
                    raw: self
                        .inner
                        .read()
                        .raw
                        .compress(
                            &TensorCompressionMethod::MinMaxUInt8(
                                MinMaxUInt8CompressionParameters {},
                            ),
                            n_chunks,
                            0,
                            target_chunk,
                        )
                        .expect("unable to compress"),
                    ready_for_comm: false,
                    ready_cuda_event_ptr: 0,
                })),
            },
            _ => {
                unimplemented!()
            }
        }
    }

    // pub fn to_numpy_f32<'py>(self_: PyRef<'py, Self>) -> pyo3::Py<PyArray1<f32>> {
    //     let inner = self_.inner.read();
    //     assert_eq!(inner.raw.dtype, BaguaTensorDtype::F32);
    //     let mut array = ndarray::Array1::from_elem((inner.raw.num_elem,), 0f32);
    //     let array_ptr = array.as_mut_ptr();
    //     let device_ptr = inner.raw.ptr;
    //     let num_bytes = inner.raw.num_elem as i32 * inner.raw.dtype.bytes() as i32;
    //     unsafe {
    //         cpp::cpp!([array_ptr as "float*", device_ptr as "float*", num_bytes as "int"]
    //         {
    //         CUDACHECK(cudaMemcpy(array_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost));
    //         });
    //     }
    //     array.into_pyarray(self_.py()).to_owned()
    // }
    //
    // pub fn to_numpy_u8<'py>(self_: PyRef<'py, Self>) -> pyo3::Py<PyArray1<u8>> {
    //     let inner = self_.inner.read();
    //     assert_eq!(inner.raw.dtype, BaguaTensorDtype::U8);
    //     let mut array = ndarray::Array1::from_elem((inner.raw.num_elem,), 0u8);
    //     let array_ptr = array.as_mut_ptr();
    //     let device_ptr = inner.raw.ptr;
    //     let num_bytes = inner.raw.num_elem as i32 * inner.raw.dtype.bytes() as i32;
    //     unsafe {
    //         cpp::cpp!([array_ptr as "uint8_t*", device_ptr as "uint8_t*", num_bytes as "int"]
    //         {
    //         CUDACHECK(cudaMemcpy(array_ptr, device_ptr, num_bytes, cudaMemcpyDeviceToHost));
    //         });
    //     }
    //     array.into_pyarray(self_.py()).to_owned()
    // }
    //
    pub fn decompress_from(&mut self, method: &str, n_chunks: usize, compressed_buffer: &Self) {
        match method {
            "min_max_uint8" => {
                self.inner.write().raw.decompress_from(
                    &TensorCompressionMethod::MinMaxUInt8(MinMaxUInt8CompressionParameters {}),
                    n_chunks,
                    &compressed_buffer.inner.read().raw,
                    0,
                );
            }
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn ptr(&self) -> u64 {
        self.inner.read().raw.ptr
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn num_elem(&self) -> usize {
        self.inner.read().raw.num_elem
    }

    pub fn num_elem_allocated(&self) -> usize {
        self.inner.read().raw.num_elem_allocated
    }

    pub fn dtype(&self) -> String {
        format!("{:?}", self.inner.read().raw.dtype)
    }

    pub fn reset_ptr(&mut self, ptr: u64) {
        self.inner.write().raw.ptr = ptr;
    }
}

#[derive(Debug)]
pub struct BaguaBucketInner {
    pub tensors: Vec<BaguaTensor>,
    pub dtype: BaguaTensorDtype,
    pub inplace: bool,
    pub comm_ops: Vec<Arc<dyn CommOpTrait + Sync + Send>>,
    pub align_bytes: usize,
}

pub struct BaguaCommunicationTensor<'b> {
    pub raw: BaguaTensorRaw,
    pub need_copy_back: bool,
    pub bucket: &'b BaguaBucketInner,
    pub stream_ptr: u64,
}

impl BaguaBucketInner {
    /// NOTE: this does not wait for memcpy finished
    pub fn get_communication_tensor(
        &self,
        stream_ptr: u64,
        force_copy: bool,
        force_not_copy_back: bool,
    ) -> BaguaCommunicationTensor {
        for tensor in &self.tensors {
            let event_ptr = { tensor.inner.read().ready_cuda_event_ptr };
            if event_ptr != 0 {
                unsafe {
                    cpp::cpp!([stream_ptr as "cudaStream_t", event_ptr as "cudaEvent_t"]
                    {
                    CUDACHECK(cudaStreamWaitEvent( stream_ptr, event_ptr , 0));
                    });
                }
            }
            tensor.inner.write().ready_cuda_event_ptr = 0;
        }
        match self.inplace && !force_copy {
            true => {
                tracing::debug!("bucket is inplace, creating communication tensor without copy");
                let tensor = self.tensors.first().unwrap().inner.read();
                let total_num_elem: usize = self
                    .tensors
                    .iter()
                    .map(|x| x.inner.read().raw.num_elem)
                    .sum();
                let total_num_elem_allocated: usize = self
                    .tensors
                    .iter()
                    .map(|x| x.inner.read().raw.num_elem_allocated)
                    .sum();
                BaguaCommunicationTensor {
                    raw: BaguaTensorRaw {
                        ptr: tensor.raw.ptr,
                        num_elem: total_num_elem,
                        dtype: tensor.raw.dtype.clone(),
                        num_elem_allocated: total_num_elem_allocated,
                        device_id: tensor.raw.device_id,
                        pool_allocation: None,
                    },
                    need_copy_back: false,
                    bucket: &self,
                    stream_ptr,
                }
            }
            false => {
                tracing::debug!("bucket is not inplace, doing buffer copy");
                let first_tensor = self.tensors.first().unwrap().inner.read();
                let total_num_elem: usize = self
                    .tensors
                    .iter()
                    .map(|x| x.inner.read().raw.num_elem)
                    .sum();
                let total_bytes = {
                    let mut result = total_num_elem * first_tensor.raw.dtype.bytes();
                    if self.align_bytes > 0 {
                        result =
                            (result + (self.align_bytes - 1)) / self.align_bytes * self.align_bytes;
                    }
                    result
                };
                assert_eq!(
                    total_bytes % first_tensor.raw.dtype.bytes(),
                    0,
                    "cannot align tensor"
                );
                let buffer_tensor = CUDA_DEVICE_MEMORY_POOL[first_tensor.raw.device_id]
                    .try_pull(total_bytes)
                    .expect("unable to allocate GPU buffer memory to do bucketing");

                let mut dst = buffer_tensor.ptr;
                for tensor in &self.tensors {
                    let tensor = tensor.inner.read();
                    let src = tensor.raw.ptr;
                    let count = tensor.raw.num_elem * tensor.raw.dtype.bytes();
                    unsafe {
                        cpp::cpp!([stream_ptr as "cudaStream_t", dst as "void *", src as "const void *", count as "size_t"]
                        {
                        CUDACHECK(cudaMemcpyAsync( dst, src , count, cudaMemcpyDeviceToDevice, stream_ptr));
                        });
                    }
                    dst += tensor.raw.num_elem as u64 * tensor.raw.dtype.bytes() as u64;
                }

                BaguaCommunicationTensor {
                    raw: BaguaTensorRaw {
                        ptr: buffer_tensor.ptr,
                        num_elem: total_num_elem,
                        dtype: first_tensor.raw.dtype.clone(),
                        num_elem_allocated: total_bytes / first_tensor.raw.dtype.bytes(),
                        device_id: first_tensor.raw.device_id,
                        pool_allocation: Some(buffer_tensor),
                    },
                    need_copy_back: if force_not_copy_back { false } else { true },
                    bucket: &self,
                    stream_ptr,
                }
            }
        }
    }
}

impl<'b> Drop for BaguaCommunicationTensor<'b> {
    fn drop(&mut self) {
        let stream_ptr = self.stream_ptr;
        if self.need_copy_back {
            let mut src = self.raw.ptr;
            for tensor in &self.bucket.tensors {
                let tensor = tensor.inner.read();
                let dst = tensor.raw.ptr;
                let count = tensor.raw.num_elem * tensor.raw.dtype.bytes();
                unsafe {
                    cpp::cpp!([stream_ptr as "cudaStream_t", dst as "void *", src as "const void *", count as "size_t"]
                    {
                    CUDACHECK(cudaMemcpyAsync( dst, src , count, cudaMemcpyDeviceToDevice, stream_ptr));
                    });
                }
                src += count as u64;
            }
        }

        unsafe {
            cpp::cpp!([stream_ptr as "cudaStream_t"]
            {
                CUDACHECK(cudaStreamSynchronize(stream_ptr));
            });
        }
        tracing::debug!("one communication finished");
    }
}

#[derive(Debug, Clone)]
pub struct BaguaBucket {
    pub id: u64,
    pub name: String,
    pub inner: Arc<Mutex<BaguaBucketInner>>,
}

impl BaguaBucket {
    pub fn new(
        tensors: &[&BaguaTensor],
        name: &str,
        inplace: bool,
        align_bytes: usize,
    ) -> Result<Self, BaguaCoreError> {
        if tensors.is_empty() {
            return Err(BaguaCoreError::BucketError("bucket is empty".into()));
        }

        let first_tensor = (*tensors.first().unwrap()).inner.read();
        let dtype: &BaguaTensorDtype = &first_tensor.raw.dtype;
        let device_id = first_tensor.raw.device_id;
        for tensor in tensors.iter() {
            let tensor = tensor.inner.read();
            if dtype != &tensor.raw.dtype {
                return Err(BaguaCoreError::BucketError(
                    "tensors in the same bucket should be of the same dtype".into(),
                ));
            }
            if device_id != tensor.raw.device_id {
                return Err(BaguaCoreError::BucketError(
                    "tensors in the same bucket should be of the same device".into(),
                ));
            }
        }

        for tensor in tensors.iter() {
            let tensor = tensor.inner.read();
            if tensor.raw.num_elem_allocated < tensor.raw.num_elem {
                return Err(BaguaCoreError::TensorError(
                    "num_elem_allocated should always be greater than num_elem in a tensor".into(),
                ));
            }
        }

        // inplace memory contiguous check
        if inplace {
            let t = &(*tensors.first().unwrap()).inner.read();
            let bytes_per_element = dtype.bytes() as u64;
            let mut current_ptr = t.raw.ptr + t.raw.num_elem_allocated as u64 * bytes_per_element;
            for tensor in tensors.iter().dropping(1) {
                let inner_tensor = &tensor.inner.read();
                if current_ptr != inner_tensor.raw.ptr {
                    return Err(BaguaCoreError::BucketError(
                        "tensors in a bucket not contiguous while marked as inplace".into(),
                    ));
                } else {
                    current_ptr += inner_tensor.raw.num_elem_allocated as u64 * bytes_per_element;
                }
            }

            let mut total_bytes = 0;
            for (i, tensor) in tensors.iter().enumerate() {
                let inner_tensor = &tensor.inner.read();
                if (inner_tensor.raw.num_elem != inner_tensor.raw.num_elem_allocated)
                    && (i != (tensors.len() - 1))
                {
                    return Err(BaguaCoreError::BucketError(
                        "non-last tensors in a bucket should have num_elem == num_elem_allocated in inplace mode".into()
                    ));
                }
                total_bytes += inner_tensor.raw.num_elem_allocated * inner_tensor.raw.dtype.bytes();
            }

            if total_bytes % align_bytes != 0 {
                return Err(BaguaCoreError::BucketError(
                    "inplace bucket tensors are not properly aligned".into(),
                ));
            }
        }

        let id = lazy_id::Id::lazy().get();
        Ok(Self {
            id,
            name: name.to_owned(),
            inner: Arc::new(Mutex::new(BaguaBucketInner {
                inplace,
                tensors: tensors.iter().map(|x| (**x).clone()).collect(),
                comm_ops: vec![],
                dtype: tensors.first().unwrap().inner.read().raw.dtype.clone(),
                align_bytes,
            })),
        })
    }

    pub fn tensors(&self) -> Vec<BaguaTensor> {
        self.inner.lock().tensors.clone()
    }

    pub fn append_decentralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicator>,
        communicator_intranode: Option<&BaguaSingleCommunicator>,
        hierarchical: bool,
        peer_selection_mode: String,
        communication_interval: usize,
        compression: Option<String>,
    ) {
        let communicator =
            BaguaCommunicator::new(communicator_internode, communicator_intranode, hierarchical)
                .expect("cannot create communicator");
        let comm_op: Arc<dyn CommOpTrait + Send + Sync> = match compression {
            None => Arc::new(DecentralizedFullPrecisionSynchronous {
                communicator,
                peer_selection_mode: match peer_selection_mode.as_str() {
                    "all" => PeerSelectionMode::All,
                    "shift_one" => PeerSelectionMode::ShiftOne,
                    &_ => {
                        unimplemented!("unsupported peer_selection_mode for decentralized algorithm (should be `all` or `shift_one`)")
                    }
                },
                step: Default::default(),
                communication_interval,
            }),
            Some(x) => match x.as_str() {
                _ => {
                    unimplemented!()
                }
            },
        };
        self.inner.lock().comm_ops.push(comm_op);
    }

    pub fn append_python_op(&mut self, op: pyo3::Py<pyo3::PyAny>) {
        let comm_op: Arc<dyn CommOpTrait + Send + Sync> = Arc::new(PythonFFIOp { py_callable: op });
        self.inner.lock().comm_ops.push(comm_op);
    }

    /// this function will use communicator_internode to communicate.
    /// if hierarchical = True, it will do hierarchical communicator, this requires intranode communicator on each node and inter node communicator on leader GPU. leader GPU will be the GPU whose communicator_intranode rank is 0
    pub fn append_centralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicator>,
        communicator_intranode: Option<&BaguaSingleCommunicator>,
        hierarchical: bool,
        average: bool,
        scattergather: bool,
        compression: Option<String>,
    ) {
        let communicator =
            BaguaCommunicator::new(communicator_internode, communicator_intranode, hierarchical)
                .expect("cannot create communicator");
        let comm_op: Arc<dyn CommOpTrait + Send + Sync> = match compression {
            None => Arc::new(CentralizedFullPrecisionSynchronous {
                communicator,
                average,
                scattergather,
            }),
            Some(x) => match x.as_str() {
                "MinMaxUInt8" => Arc::new(CentralizedLowPrecisionSynchronous {
                    communicator,
                    average,
                    compression_method: TensorCompressionMethod::MinMaxUInt8(
                        MinMaxUInt8CompressionParameters {},
                    ),
                }),
                _ => {
                    unimplemented!()
                }
            },
        };
        self.inner.lock().comm_ops.push(comm_op);
    }

    pub fn ready_for_comm(&self) -> bool {
        self.inner.lock().tensors.iter().all(|x| x.ready_for_comm())
    }

    pub fn reset_comm_ready(&self) {
        self.inner
            .lock()
            .tensors
            .iter()
            .for_each(|x| x.mark_comm_not_ready());
    }
}
