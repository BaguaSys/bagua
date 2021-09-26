use crate::comm_ops::centralized_full_precision_synchronous::CentralizedFullPrecisionSynchronous;
use crate::comm_ops::centralized_low_precision_synchronous::CentralizedLowPrecisionSynchronous;
use crate::comm_ops::decentralized_full_precision_asynchronous::DecentralizedFullPrecisionAsynchronous;
use crate::comm_ops::decentralized_full_precision_synchronous::{
    DecentralizedFullPrecisionSynchronous, PeerSelectionMode,
};
use crate::comm_ops::decentralized_low_precision_synchronous::DecentralizedLowPrecisionSynchronous;
use crate::comm_ops::python_ffi_op::PythonFFIOp;
use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaSingleCommunicator};
use crate::resource_pool::{CudaMemory, CUDA_DEVICE_MEMORY_POOL};
use crate::torch_ffi::root::c10::{DeviceType, StorageImpl, TensorImpl};
use crate::{kernels, BaguaCoreError};
use itertools::Itertools;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use parking_lot::{Mutex, RwLock};
use pyo3::types::IntoPyDict;
use sized_object_pool::DynamicPoolItem;
use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

// must be consistent with Aluminum ReductionOperator: https://github.com/BaguaSys/Aluminum/blob/master/include/aluminum/base.hpp
#[derive(Clone, Copy, Debug, PartialEq, FromPrimitive)]
pub enum BaguaReductionOp {
    SUM,
    PROD,
    MIN,
    MAX,
    LOR,
    LAND,
    LXOR,
    BOR,
    BAND,
    BXOR,
    AVG,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BaguaTensorDtype {
    F32,
    F16,
    U8,
    I64,
    U64,
}

impl BaguaTensorDtype {
    pub fn bytes(&self) -> usize {
        match self {
            BaguaTensorDtype::F32 => 4,
            BaguaTensorDtype::F16 => 2,
            BaguaTensorDtype::U8 => 1,
            BaguaTensorDtype::I64 => 8,
            BaguaTensorDtype::U64 => 8,
        }
    }

    pub fn to_nccl_datatype(&self) -> i32 {
        match self {
            BaguaTensorDtype::F32 => 7,
            BaguaTensorDtype::F16 => 6,
            BaguaTensorDtype::U8 => 1,
            BaguaTensorDtype::I64 => 4,
            BaguaTensorDtype::U64 => 5,
        }
    }
}

#[derive(Debug)]
pub struct TorchTensorRaw {
    pub torch_tensor_cdata: u64,
    pub python_fallback: bool,
    pub dtype: BaguaTensorDtype,
}

#[derive(Debug)]
pub struct BaguaTensorRaw {
    pub ptr: u64,
    pub num_elem_allocated: usize,
    pub dtype: BaguaTensorDtype,
    pub num_elem: usize,
    pub device_id: usize,
    pub pool_allocations: Vec<Arc<DynamicPoolItem<CudaMemory>>>,
}

pub trait RawBaguaTensor: Debug {
    fn data_ptr(&self) -> u64;
    fn num_elements(&self) -> usize;
    fn num_elements_allocated(&self) -> usize;
    fn device_id(&self) -> usize;
    fn dtype(&self) -> BaguaTensorDtype;

    fn divide_inplace(&mut self, stream_ptr: u64, divide_factor: f32) {
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();
        unsafe {
            match self.dtype() {
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
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            }
        }
    }

    fn clone_from(&mut self, other: &dyn RawBaguaTensor, stream_ptr: u64) {
        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.num_elements(), other.num_elements());
        unsafe {
            let src = other.data_ptr();
            let dst = self.data_ptr();
            let count = self.num_elements() * self.dtype().bytes();
            cpp::cpp!([stream_ptr as "cudaStream_t", dst as "void *", src as "const void *", count as "size_t"]
            {
            CUDACHECK(cudaMemcpyAsync( dst, src , count, cudaMemcpyDeviceToDevice, stream_ptr));
            });
        }
    }

    fn async_model_average(
        &mut self,
        reduced_tensor: &dyn RawBaguaTensor,
        tensor: &dyn RawBaguaTensor,
        nranks: f32,
        stream_ptr: u64,
    ) {
        assert_eq!(self.dtype(), reduced_tensor.dtype());
        assert_eq!(self.num_elements(), reduced_tensor.num_elements());
        assert_eq!(self.dtype(), tensor.dtype());
        assert_eq!(self.num_elements(), tensor.num_elements());
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();
        unsafe {
            kernels::async_model_average_host(
                tensor_ptr as _,
                reduced_tensor.data_ptr() as _,
                tensor.data_ptr() as _,
                nranks as f32,
                total_num_elem as i32,
                stream_ptr as _,
            );
        }
    }

    fn substract_inplace(&mut self, other: &dyn RawBaguaTensor, stream_ptr: u64) {
        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.num_elements(), other.num_elements());
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();
        unsafe {
            match self.dtype() {
                BaguaTensorDtype::F32 => {
                    kernels::substract_inplace_f32_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::substract_inplace_f16_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
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
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            }
        }
    }

    fn add_inplace(&mut self, other: &dyn RawBaguaTensor, stream_ptr: u64) {
        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.num_elements(), other.num_elements());
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();
        unsafe {
            match self.dtype() {
                BaguaTensorDtype::F32 => {
                    kernels::add_inplace_f32_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::add_inplace_f16_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
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
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            }
        }
    }
    fn addmul_inplace(&mut self, other: &dyn RawBaguaTensor, factor: f32, stream_ptr: u64) {
        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.num_elements(), other.num_elements());
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();
        unsafe {
            match self.dtype() {
                BaguaTensorDtype::F32 => {
                    kernels::addmul_inplace_f32_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
                        total_num_elem as i32,
                        factor as _,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::addmul_inplace_f16_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
                        total_num_elem as i32,
                        factor as _,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::U8 => {
                    unimplemented!()
                }
                BaguaTensorDtype::I64 => {
                    unimplemented!()
                }
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            }
        }
    }

    fn average_inplace(&mut self, other: &dyn RawBaguaTensor, stream_ptr: u64) {
        assert_eq!(self.dtype(), other.dtype());
        assert_eq!(self.num_elements(), other.num_elements());
        let tensor_ptr = self.data_ptr();
        let total_num_elem = self.num_elements();

        unsafe {
            match self.dtype() {
                BaguaTensorDtype::F32 => {
                    kernels::average_inplace_f32_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
                        total_num_elem as i32,
                        stream_ptr as _,
                    );
                }
                BaguaTensorDtype::F16 => {
                    kernels::average_inplace_f16_host(
                        tensor_ptr as _,
                        other.data_ptr() as _,
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
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            }
        }
    }

    fn compress(
        &self,
        compression: &TensorCompressionMethod,
        n_chunks: usize,
        stream_ptr: u64,
        target_chunk: i32,
    ) -> Result<Box<dyn RawBaguaTensor + Send + Sync>, BaguaCoreError> {
        match compression {
            TensorCompressionMethod::MinMaxUInt8(_parameters) => {
                assert_eq!(
                    self.num_elements_allocated() % n_chunks,
                    0,
                    "compression tensor size % n_chunks must be 0"
                );
                let chunk_size = self.num_elements_allocated() / n_chunks;
                let output_buffer_size =
                    MinMaxUInt8CompressionParameters::get_compressed_buffer_size(
                        n_chunks,
                        chunk_size,
                        &self.dtype(),
                    );
                let output_buffer =
                    CUDA_DEVICE_MEMORY_POOL[self.device_id()].try_pull(output_buffer_size)?;

                let temp_buffer_size = MinMaxUInt8CompressionParameters::get_temp_buffer_size(
                    self.data_ptr(),
                    self.num_elements(),
                    output_buffer.ptr,
                    stream_ptr,
                    &self.dtype(),
                );
                let temp_buffer =
                    CUDA_DEVICE_MEMORY_POOL[self.device_id()].try_pull(temp_buffer_size)?;

                match self.dtype() {
                    BaguaTensorDtype::F32 => unsafe {
                        kernels::compress_f32_to_uint8_host(
                            self.data_ptr() as _,
                            self.num_elements() as _,
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
                            self.data_ptr() as _,
                            self.num_elements() as _,
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
                    BaguaTensorDtype::U64 => {
                        unimplemented!()
                    }
                }
                return Ok(Box::new(BaguaTensorRaw {
                    ptr: output_buffer.ptr,
                    num_elem_allocated: output_buffer_size,
                    dtype: BaguaTensorDtype::U8,
                    num_elem: output_buffer_size,
                    device_id: self.device_id(),
                    pool_allocations: vec![Arc::new(output_buffer)],
                }));
            }
        }
    }

    fn decompress_from(
        &mut self,
        compression: &TensorCompressionMethod,
        n_chunks: usize,
        compressed_buffer: &dyn RawBaguaTensor,
        stream_ptr: u64,
    ) {
        assert_eq!(
            self.num_elements_allocated() % n_chunks,
            0,
            "compression tensor size % n_chunks must be 0"
        );
        let chunk_size = self.num_elements_allocated() / n_chunks;
        match compression {
            TensorCompressionMethod::MinMaxUInt8(_parameters) => match self.dtype() {
                BaguaTensorDtype::F32 => unsafe {
                    kernels::decompress_uint8_to_f32_host(
                        compressed_buffer.data_ptr() as _,
                        compressed_buffer.num_elements_allocated()
                            * compressed_buffer.dtype().bytes(),
                        chunk_size as _,
                        n_chunks as _,
                        self.data_ptr() as _,
                        stream_ptr as _,
                    );
                },
                BaguaTensorDtype::F16 => unsafe {
                    kernels::decompress_uint8_to_f16_host(
                        compressed_buffer.data_ptr() as _,
                        compressed_buffer.num_elements_allocated()
                            * compressed_buffer.dtype().bytes(),
                        chunk_size as _,
                        n_chunks as _,
                        self.data_ptr() as _,
                        stream_ptr as _,
                    );
                },
                BaguaTensorDtype::U8 => {
                    unimplemented!()
                }
                BaguaTensorDtype::I64 => {
                    unimplemented!()
                }
                BaguaTensorDtype::U64 => {
                    unimplemented!()
                }
            },
        }
    }

    fn reduce_mean_inplace(&mut self, n_chunks: usize, target_chunk: usize, stream_ptr: u64) {
        assert_eq!(
            self.num_elements_allocated() % n_chunks,
            0,
            "reduce_mean_inplace requires tensor aligned"
        );
        let chunk_size = self.num_elements_allocated() / n_chunks;
        match self.dtype() {
            BaguaTensorDtype::F32 => unsafe {
                kernels::reduce_mean_f32_inplace_host(
                    self.data_ptr() as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::F16 => unsafe {
                kernels::reduce_mean_f16_inplace_host(
                    self.data_ptr() as _,
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
            BaguaTensorDtype::U64 => {
                unimplemented!()
            }
        }
    }

    fn reduce_sum_inplace(&mut self, n_chunks: usize, target_chunk: usize, stream_ptr: u64) {
        assert_eq!(
            self.num_elements_allocated() % n_chunks,
            0,
            "reduce_sum_inplace requires tensor aligned"
        );
        let chunk_size = self.num_elements_allocated() / n_chunks;
        match self.dtype() {
            BaguaTensorDtype::F32 => unsafe {
                kernels::reduce_sum_f32_inplace_host(
                    self.data_ptr() as _,
                    chunk_size as _,
                    n_chunks as _,
                    target_chunk as _,
                    stream_ptr as _,
                );
            },
            BaguaTensorDtype::F16 => unsafe {
                kernels::reduce_sum_f16_inplace_host(
                    self.data_ptr() as _,
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
            BaguaTensorDtype::U64 => {
                unimplemented!()
            }
        }
    }
}

impl TorchTensorRaw {
    pub fn get_pytensor<'a>(cdata: u64, py: &'a pyo3::Python) -> pyo3::PyResult<&'a pyo3::PyAny> {
        let torch = py.import("torch")?;
        let tensor_class = torch.get("Tensor")?;
        tensor_class.call((), Some([("cdata", cdata)].into_py_dict(py.to_owned())))
    }

    pub fn check_consistency_with_python(&self) -> pyo3::PyResult<bool> {
        fn check_consistency(
            py: pyo3::Python,
            cdata: u64,
            data_ptr: u64,
            numel: usize,
            device_id: usize,
        ) -> pyo3::PyResult<bool> {
            let py_tensor = TorchTensorRaw::get_pytensor(cdata, &py)?;
            let py_data_ptr: u64 = py_tensor.call_method0("data_ptr")?.extract()?;
            if py_data_ptr != data_ptr {
                return Ok(false);
            }
            let py_numel: usize = py_tensor.call_method0("numel")?.extract()?;
            if py_numel != numel {
                return Ok(false);
            }
            let py_device_id: usize = py_tensor.getattr("device")?.getattr("index")?.extract()?;
            if py_device_id != device_id {
                return Ok(false);
            }
            return Ok(true);
        }

        pyo3::Python::with_gil(|py| {
            check_consistency(
                py,
                self.torch_tensor_cdata,
                self.data_ptr(),
                self.num_elements(),
                self.device_id(),
            )
        })
    }

    fn extract_torch_c_data(&self) -> &TensorImpl {
        unsafe { (self.torch_tensor_cdata as *const TensorImpl).as_ref() }
            .expect("torch c data pointer is null")
    }

    fn extract_storage(&self) -> &StorageImpl {
        unsafe {
            (self.extract_torch_c_data().storage_.storage_impl_.target_ as *const StorageImpl)
                .as_ref()
                .expect("torch c data has no storage")
        }
    }
}

impl RawBaguaTensor for TorchTensorRaw {
    fn data_ptr(&self) -> u64 {
        if self.python_fallback {
            pyo3::Python::with_gil(|py| {
                let py_tensor = TorchTensorRaw::get_pytensor(self.torch_tensor_cdata, &py).unwrap();
                py_tensor
                    .call_method0("data_ptr")
                    .unwrap()
                    .extract()
                    .unwrap()
            })
        } else {
            let cdata = self.extract_torch_c_data();
            let storage = self.extract_storage();
            storage.data_ptr_.ptr_.data_ as u64
                + cdata.storage_offset_ as u64 * self.dtype.bytes() as u64
        }
    }

    fn num_elements(&self) -> usize {
        if self.python_fallback {
            pyo3::Python::with_gil(|py| {
                let py_tensor = TorchTensorRaw::get_pytensor(self.torch_tensor_cdata, &py).unwrap();
                py_tensor.call_method0("numel").unwrap().extract().unwrap()
            })
        } else {
            self.extract_torch_c_data().numel_ as _
        }
    }

    fn num_elements_allocated(&self) -> usize {
        self.num_elements()
    }

    fn device_id(&self) -> usize {
        if self.python_fallback {
            pyo3::Python::with_gil(|py| {
                let py_tensor = TorchTensorRaw::get_pytensor(self.torch_tensor_cdata, &py).unwrap();
                py_tensor
                    .getattr("device")
                    .unwrap()
                    .getattr("index")
                    .unwrap()
                    .extract()
                    .unwrap()
            })
        } else {
            let storage_data_ptr = &self.extract_storage().data_ptr_;
            assert_eq!(
                storage_data_ptr.device_.type_,
                DeviceType::CUDA,
                "currently only cuda tensors are supported in Bagua"
            );
            return storage_data_ptr.device_.index_ as _;
        }
    }

    fn dtype(&self) -> BaguaTensorDtype {
        self.dtype
    }
}

impl RawBaguaTensor for BaguaTensorRaw {
    fn data_ptr(&self) -> u64 {
        self.ptr
    }

    fn num_elements(&self) -> usize {
        self.num_elem
    }

    fn num_elements_allocated(&self) -> usize {
        self.num_elem_allocated
    }

    fn device_id(&self) -> usize {
        self.device_id
    }

    fn dtype(&self) -> BaguaTensorDtype {
        self.dtype
    }
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
                let compressed_align_bytes = align_size(chunk_size, align_bytes) * n_chunks;
                let min_max_align_bytes = align_size(4 * 2, align_bytes) * n_chunks; // assume min max are both f32
                return compressed_align_bytes + min_max_align_bytes;
            }
            BaguaTensorDtype::F16 => {
                let align_bytes = 32;
                // Note chunk_size is already aligned outside
                let compressed_align_bytes = align_size(chunk_size, align_bytes) * n_chunks;
                let min_max_align_bytes = align_size(2 * 2, align_bytes) * n_chunks; // assume min max are both f16
                return compressed_align_bytes + min_max_align_bytes;
            }
            BaguaTensorDtype::U8 => {
                unimplemented!()
            }
            BaguaTensorDtype::I64 => {
                unimplemented!()
            }
            BaguaTensorDtype::U64 => {
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
            BaguaTensorDtype::U64 => {
                unimplemented!()
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum TensorCompressionMethod {
    MinMaxUInt8(MinMaxUInt8CompressionParameters),
}

impl BaguaTensorRaw {}

#[derive(Debug)]
pub struct BaguaTensorInner {
    pub name: String,
    pub raw: Box<dyn RawBaguaTensor + Send + Sync>,
    pub ready_for_comm: bool,
    pub ready_cuda_event_ptr: u64,
}

#[derive(Clone, Debug)]
pub struct BaguaTensor {
    pub inner: Arc<RwLock<BaguaTensorInner>>,
}

impl BaguaTensor {
    pub fn new_from_torch(
        name: String,
        torch_cdata_ptr: u64,
        dtype: BaguaTensorDtype,
    ) -> pyo3::PyResult<Self> {
        let mut torch_tensor = TorchTensorRaw {
            torch_tensor_cdata: torch_cdata_ptr,
            python_fallback: false,
            dtype,
        };
        let consistency = torch_tensor.check_consistency_with_python()?;
        if !consistency {
            tracing::warn!(
                r#"PyTorch tensor memory layout inconsistent with latest PyTorch. Bagua will fallback to Python interface. This will degrade system performance. We suggest upgrading to latest PyTorch."#
            )
        }
        torch_tensor.python_fallback = !consistency;

        Ok(Self {
            inner: Arc::new(RwLock::new(BaguaTensorInner {
                name,
                raw: Box::new(torch_tensor),
                ready_for_comm: false,
                ready_cuda_event_ptr: 0,
            })),
        })
    }

    pub fn mark_comm_ready(&self, cuda_event_ptr: u64) {
        if cuda_event_ptr == 0 {
            tracing::info!("mark comm ready with an event 0, ignoring event");
        }
        let mut guard = self.inner.write();
        guard.ready_for_comm = true;
        guard.ready_cuda_event_ptr = cuda_event_ptr;
    }

    pub fn mark_comm_not_ready(&self) {
        self.inner.write().ready_for_comm = false;
    }

    pub fn name(&self) -> String {
        self.inner.read().name.clone()
    }

    pub fn ready_for_comm(&self) -> bool {
        let inner = self.inner.read();
        inner.ready_for_comm || inner.name.starts_with("bagua_padding_tensor")
    }

    pub fn compress(&self, method: &str, n_chunks: usize, target_chunk: i32) -> Self {
        match method {
            "MinMaxUInt8" => Self {
                inner: Arc::new(RwLock::new(BaguaTensorInner {
                    name: "compressed_tensor".to_string(),
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
                        .expect("cannot compress tensor"),
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
            "MinMaxUInt8" => {
                self.inner.write().raw.decompress_from(
                    &TensorCompressionMethod::MinMaxUInt8(MinMaxUInt8CompressionParameters {}),
                    n_chunks,
                    compressed_buffer.inner.read().raw.as_ref(),
                    0,
                );
            }
            _ => {
                unimplemented!()
            }
        }
    }

    pub fn data_ptr(&self) -> u64 {
        self.inner.read().raw.data_ptr()
    }

    pub fn device_id(&self) -> usize {
        self.inner.read().raw.device_id()
    }

    pub fn num_elements(&self) -> usize {
        self.inner.read().raw.num_elements()
    }

    pub fn num_elements_allocated(&self) -> usize {
        self.inner.read().raw.num_elements_allocated()
    }

    pub fn dtype(&self) -> String {
        format!("{:?}", self.inner.read().raw.dtype())
    }
}

#[derive(Debug)]
pub struct BaguaBucketInner {
    pub tensors: Vec<BaguaTensor>,
    pub dtype: BaguaTensorDtype,
    pub comm_ops: Vec<Arc<dyn CommOpTrait + Sync + Send>>,
}

pub struct BaguaCommunicationTensor<'b> {
    pub raw: BaguaTensorRaw,
    pub need_copy_back: bool,
    pub bucket: &'b BaguaBucketInner,
    pub stream_ptr: u64,
}

impl BaguaBucketInner {
    pub fn contiguous(&self) -> bool {
        let bytes_per_element = self.dtype.bytes() as u64;
        let t = &(*self.tensors.first().unwrap()).inner.read();
        let mut current_ptr =
            t.raw.data_ptr() + t.raw.num_elements_allocated() as u64 * bytes_per_element;
        for tensor in self.tensors.iter().dropping(1) {
            let inner_tensor = &tensor.inner.read();
            if current_ptr != inner_tensor.raw.data_ptr() {
                return false;
            } else {
                current_ptr += inner_tensor.raw.num_elements_allocated() as u64 * bytes_per_element;
            }
        }
        return true;
    }

    pub fn total_num_elements(&self) -> usize {
        self.tensors
            .iter()
            .map(|tensor| tensor.num_elements())
            .sum::<usize>()
    }

    pub fn total_num_elements_allocated(&self) -> usize {
        self.tensors
            .iter()
            .map(|tensor| tensor.num_elements_allocated())
            .sum::<usize>()
    }

    pub fn total_allocated_bytes(&self) -> usize {
        self.total_num_elements_allocated() * self.dtype.bytes()
    }

    /// NOTE: this does not wait for memcpy finished
    // TODO: simplify args
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
        match self.contiguous() && !force_copy {
            true => {
                tracing::debug!("bucket is inplace, creating communication tensor without copy");
                let total_num_elem_allocated: usize = self.total_num_elements_allocated();
                BaguaCommunicationTensor {
                    raw: BaguaTensorRaw {
                        ptr: self.tensors[0].data_ptr(),
                        num_elem: total_num_elem_allocated,
                        dtype: self.dtype,
                        num_elem_allocated: total_num_elem_allocated,
                        device_id: self.tensors[0].device_id(),
                        pool_allocations: vec![],
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
                    .map(|x| x.inner.read().raw.num_elements())
                    .sum();
                let buffer_tensor = CUDA_DEVICE_MEMORY_POOL[first_tensor.raw.device_id()]
                    .try_pull(self.total_allocated_bytes())
                    .expect("unable to allocate GPU buffer memory to do bucketing");

                let mut dst = buffer_tensor.ptr;
                for tensor in &self.tensors {
                    let tensor = tensor.inner.read();
                    let src = tensor.raw.data_ptr();
                    let count = tensor.raw.num_elements() * tensor.raw.dtype().bytes();
                    unsafe {
                        cpp::cpp!([stream_ptr as "cudaStream_t", dst as "void *", src as "const void *", count as "size_t"]
                        {
                        CUDACHECK(cudaMemcpyAsync( dst, src , count, cudaMemcpyDeviceToDevice, stream_ptr));
                        });
                    }
                    dst += tensor.raw.num_elements() as u64 * tensor.raw.dtype().bytes() as u64;
                }

                BaguaCommunicationTensor {
                    raw: BaguaTensorRaw {
                        ptr: buffer_tensor.ptr,
                        num_elem: total_num_elem,
                        dtype: first_tensor.raw.dtype().clone(),
                        num_elem_allocated: self.total_num_elements_allocated(),
                        device_id: first_tensor.raw.device_id(),
                        pool_allocations: vec![Arc::new(buffer_tensor)],
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
                let dst = tensor.raw.data_ptr();
                let count = tensor.raw.num_elements() * tensor.raw.dtype().bytes();
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
    pub name: String,
    pub inner: Arc<Mutex<BaguaBucketInner>>,
}

impl BaguaBucket {
    pub fn new(tensors: &[&BaguaTensor], name: &str) -> Result<Self, BaguaCoreError> {
        if tensors.is_empty() {
            return Err(BaguaCoreError::BucketError("bucket is empty".into()));
        }

        let first_tensor = (*tensors.first().unwrap()).inner.read();
        let dtype: &BaguaTensorDtype = &first_tensor.raw.dtype();
        let device_id = first_tensor.raw.device_id();
        for tensor in tensors.iter() {
            let tensor = tensor.inner.read();
            if dtype != &tensor.raw.dtype() {
                return Err(BaguaCoreError::BucketError(
                    "tensors in the same bucket should be of the same dtype".into(),
                ));
            }
            if device_id != tensor.raw.device_id() {
                return Err(BaguaCoreError::BucketError(
                    "tensors in the same bucket should be of the same device".into(),
                ));
            }
        }

        for tensor in tensors.iter() {
            let tensor = tensor.inner.read();
            if tensor.raw.num_elements_allocated() < tensor.raw.num_elements() {
                return Err(BaguaCoreError::TensorError(
                    "num_elem_allocated should always be greater than num_elem in a tensor".into(),
                ));
            }
        }

        Ok(Self {
            name: name.to_owned(),
            inner: Arc::new(Mutex::new(BaguaBucketInner {
                tensors: tensors.iter().map(|x| (**x).clone()).collect(),
                comm_ops: vec![],
                dtype: tensors.first().unwrap().inner.read().raw.dtype().clone(),
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
        peer_weight: BaguaTensor,
    ) {
        let communicator =
            BaguaCommunicator::new(communicator_internode, communicator_intranode, hierarchical)
                .expect("cannot create communicator");
        let comm_op: Arc<dyn CommOpTrait + Send + Sync> = Arc::new(
            DecentralizedFullPrecisionSynchronous {
                communicator,
                peer_selection_mode: match peer_selection_mode.as_str() {
                    "all" => PeerSelectionMode::All,
                    "shift_one" => PeerSelectionMode::ShiftOne,
                    &_ => {
                        unimplemented!("unsupported peer_selection_mode for decentralized algorithm (should be `all` or `shift_one`)")
                    }
                },
                step: Default::default(),
                peer_weight,
            },
        );

        self.inner.lock().comm_ops.push(comm_op);
    }

    pub fn append_low_precision_decentralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicator>,
        communicator_intranode: Option<&BaguaSingleCommunicator>,
        hierarchical: bool,
        peer_selection_mode: String,
        compression: String,
        weight: BaguaTensor,
        left_peer_weight: BaguaTensor,
        right_peer_weight: BaguaTensor,
    ) {
        let communicator =
            BaguaCommunicator::new(communicator_internode, communicator_intranode, hierarchical)
                .expect("cannot create communicator");
        let comm_op: Arc<dyn CommOpTrait + Send + Sync> = Arc::new(
            DecentralizedLowPrecisionSynchronous {
                communicator,
                peer_selection_mode: match peer_selection_mode.as_str() {
                    "ring" => PeerSelectionMode::Ring,
                    &_ => {
                        unimplemented!("unsupported peer_selection_mode for low precision decentralized algorithm (should be `ring`)")
                    }
                },
                compression_method: TensorCompressionMethod::MinMaxUInt8(
                    MinMaxUInt8CompressionParameters {},
                ),
                weight,
                left_peer_weight,
                right_peer_weight,
            },
        );

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

    pub fn append_decentralized_asynchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicator>,
        communicator_intranode: Option<&BaguaSingleCommunicator>,
        peer_selection_mode: String,
        torch_stream: u64,
    ) -> Arc<DecentralizedFullPrecisionAsynchronous> {
        let communicator =
            BaguaCommunicator::new(communicator_internode, communicator_intranode, false)
                .expect("cannot create communicator");

        let comm_op = Arc::new(DecentralizedFullPrecisionAsynchronous {
            communicator,
            peer_selection_mode: match peer_selection_mode.as_str() {
                "all" => PeerSelectionMode::All,
                &_ => {
                    unimplemented!("unsupported peer_selection_mode for decentralized asynchronous algorithm (should be `all`)")
                }
            },
            torch_stream,
            weight_mutex: Arc::new(Mutex::new(true)),
        });

        self.inner
            .lock()
            .comm_ops
            .push(comm_op.clone() as Arc<dyn CommOpTrait + Send + Sync>);

        comm_op
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
