#![allow(clippy::needless_return)]

use bagua_core_internal::comm_ops::decentralized_full_precision_asynchronous::DecentralizedFullPrecisionAsynchronous;
use bagua_core_internal::communicators::BaguaSingleCommunicator;
use bagua_core_internal::datatypes::{
    BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorDtype,
};
use bagua_core_internal::BaguaCommBackend;
use num_traits::FromPrimitive;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::PyNativeType;
use std::sync::Arc;

#[pyclass(dict)]
pub struct BaguaSingleCommunicatorPy {
    inner: BaguaSingleCommunicator,
}

#[pymethods]
impl BaguaSingleCommunicatorPy {
    #[new]
    pub fn new(
        rank: usize,
        nranks: usize,
        device_id: usize,
        stream_ptr: u64,
        nccl_unique_id_str: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: bagua_core_internal::communicators::BaguaSingleCommunicator::new(
                rank,
                nranks,
                device_id,
                stream_ptr,
                nccl_unique_id_str,
            ),
        })
    }

    pub fn nranks(&self) -> usize {
        self.inner.nranks()
    }

    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    pub fn device_id(&self) -> usize {
        self.inner.device_id()
    }

    pub fn abort(&self) {
        self.inner.abort()
    }

    pub fn check_abort(&self) -> bool {
        self.inner.check_abort()
    }

    pub fn allreduce(&self, send_tensor: &BaguaTensorPy, recv_tensor: &mut BaguaTensorPy, op: u8) {
        self.inner.allreduce(
            &send_tensor.inner,
            &mut recv_tensor.inner,
            BaguaReductionOp::from_u8(op).unwrap(),
        )
    }

    pub fn allreduce_inplace(&self, tensor: &mut BaguaTensorPy, op: u8) {
        self.inner
            .allreduce_inplace(&mut tensor.inner, BaguaReductionOp::from_u8(op).unwrap())
    }

    pub fn broadcast(&self, tensor: &mut BaguaTensorPy, root_rank: i32) {
        self.inner.broadcast(&mut tensor.inner, root_rank)
    }

    pub fn reduce(
        &self,
        send_tensor: &BaguaTensorPy,
        recv_tensor: &mut BaguaTensorPy,
        dst: i32,
        op: u8,
    ) {
        self.inner.reduce(
            &send_tensor.inner,
            &mut recv_tensor.inner,
            dst,
            BaguaReductionOp::from_u8(op).unwrap(),
        )
    }

    pub fn reduce_inplace(&self, tensor: &mut BaguaTensorPy, dst: i32, op: u8) {
        self.inner.reduce_inplace(
            &mut tensor.inner,
            dst,
            BaguaReductionOp::from_u8(op).unwrap(),
        )
    }

    pub fn send(&self, tensor: &mut BaguaTensorPy, peer_rank: i32) {
        self.inner.send(&mut tensor.inner, peer_rank)
    }

    pub fn recv(&self, tensor: &mut BaguaTensorPy, peer_rank: i32) {
        self.inner.recv(&mut tensor.inner, peer_rank)
    }

    pub fn alltoall(&self, send_tensor: &BaguaTensorPy, recv_tensor: &mut BaguaTensorPy) {
        self.inner
            .alltoall(&send_tensor.inner, &mut recv_tensor.inner)
    }

    pub fn alltoall_inplace(&self, tensor: &mut BaguaTensorPy) {
        self.inner.alltoall_inplace(&mut tensor.inner)
    }

    pub fn alltoall_v(
        &self,
        send_tensor: &mut BaguaTensorPy,
        send_counts: Vec<usize>,
        send_displs: Vec<usize>,
        recv_tensor: &mut BaguaTensorPy,
        recv_counts: Vec<usize>,
        recv_displs: Vec<usize>,
    ) {
        self.inner.alltoall_v(
            &mut send_tensor.inner,
            &send_counts,
            &send_displs,
            &mut recv_tensor.inner,
            &recv_counts,
            &recv_displs,
        )
    }

    pub fn allgather(&self, send_tensor: &BaguaTensorPy, recv_tensor: &mut BaguaTensorPy) {
        self.inner
            .allgather(&send_tensor.inner, &mut recv_tensor.inner)
    }

    pub fn allgather_inplace(&self, tensor: &mut BaguaTensorPy) {
        self.inner.allgather_inplace(&mut tensor.inner)
    }

    pub fn gather(
        &self,
        send_tensor: &mut BaguaTensorPy,
        recv_tensor: &mut BaguaTensorPy,
        dst: i32,
    ) {
        self.inner
            .gather(&mut send_tensor.inner, &mut recv_tensor.inner, dst)
    }

    pub fn gather_inplace(&self, tensor: &mut BaguaTensorPy, count: usize, dst: i32) {
        self.inner.gather_inplace(&mut tensor.inner, count, dst)
    }

    pub fn scatter(&self, send_tensor: &BaguaTensorPy, recv_tensor: &mut BaguaTensorPy, src: i32) {
        self.inner
            .scatter(&send_tensor.inner, &mut recv_tensor.inner, src)
    }

    pub fn scatter_inplace(&self, tensor: &mut BaguaTensorPy, count: usize, src: i32) {
        self.inner.scatter_inplace(&mut tensor.inner, count, src)
    }

    pub fn reduce_scatter(
        &self,
        send_tensor: &BaguaTensorPy,
        recv_tensor: &mut BaguaTensorPy,
        op: u8,
    ) {
        self.inner.reduce_scatter(
            &send_tensor.inner,
            &mut recv_tensor.inner,
            BaguaReductionOp::from_u8(op).unwrap(),
        )
    }

    pub fn reduce_scatter_inplace(&self, tensor: &mut BaguaTensorPy, op: u8) {
        self.inner
            .reduce_scatter_inplace(&mut tensor.inner, BaguaReductionOp::from_u8(op).unwrap())
    }

    pub fn barrier(&self) {
        self.inner.barrier()
    }

    #[staticmethod]
    pub fn generate_nccl_unique_id_str() -> String {
        BaguaSingleCommunicator::generate_nccl_unique_id_str()
    }
}

#[pyclass(dict)]
pub struct DecentralizedFullPrecisionAsynchronousPy {
    inner: Arc<DecentralizedFullPrecisionAsynchronous>,
}

#[pymethods]
impl DecentralizedFullPrecisionAsynchronousPy {
    pub fn lock_weight(&self) {
        self.inner.lock_weight()
    }

    pub fn unlock_weight(&self) {
        self.inner.unlock_weight()
    }
}

#[pyclass(dict)]
pub struct BaguaTensorPy {
    inner: BaguaTensor,
}

#[pymethods]
impl BaguaTensorPy {
    #[new]
    pub fn new(torch_tensor: &PyAny, name: String) -> PyResult<Self> {
        // TODO: sanity check
        let dtype = torch_tensor
            .getattr("dtype")
            .expect("must pass valid torch tensor")
            .repr()?
            .to_string();
        let bagua_dtype = match dtype.as_str() {
            "torch.float32" => BaguaTensorDtype::F32,
            "torch.float16" => BaguaTensorDtype::F16,
            "torch.int64" => BaguaTensorDtype::I64,
            "torch.uint8" => BaguaTensorDtype::U8,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported tensor dtype {}",
                    dtype
                )))
            }
        };
        Ok(Self {
            inner: BaguaTensor::new_from_torch(
                name,
                torch_tensor
                    .getattr("_cdata")
                    .expect("must pass valid torch tensor")
                    .extract()?,
                bagua_dtype,
            )?,
        })
    }

    pub fn compress(&self, method: &str, n_chunks: usize, target_chunk: i32) -> Self {
        Self {
            inner: self.inner.compress(method, n_chunks, target_chunk),
        }
    }

    pub fn to_numpy_f32<'py>(self_: PyRef<'py, Self>) -> pyo3::Py<PyArray1<f32>> {
        let inner = self_.inner.inner.read();
        assert_eq!(inner.raw.dtype(), BaguaTensorDtype::F32);
        let mut array = ndarray::Array1::from_elem((inner.raw.num_elements(),), 0f32);
        let array_ptr = array.as_mut_ptr();
        let device_ptr = inner.raw.data_ptr();
        let num_bytes = inner.raw.num_elements() as i32 * inner.raw.dtype().bytes() as i32;
        unsafe {
            bagua_core_internal::cuda_utils::cuda_memcpy_device_to_host_sync(
                array_ptr as _,
                device_ptr as _,
                num_bytes,
            );
        }
        array.into_pyarray(self_.py()).to_owned()
    }

    pub fn to_numpy_u8<'py>(self_: PyRef<'py, Self>) -> pyo3::Py<PyArray1<u8>> {
        let inner = self_.inner.inner.read();
        assert_eq!(inner.raw.dtype(), BaguaTensorDtype::U8);
        let mut array = ndarray::Array1::from_elem((inner.raw.num_elements(),), 0u8);
        let array_ptr = array.as_mut_ptr();
        let device_ptr = inner.raw.data_ptr();
        let num_bytes = inner.raw.num_elements() as i32 * inner.raw.dtype().bytes() as i32;
        unsafe {
            bagua_core_internal::cuda_utils::cuda_memcpy_device_to_host_sync(
                array_ptr as _,
                device_ptr as _,
                num_bytes,
            );
        }
        array.into_pyarray(self_.py()).to_owned()
    }

    pub fn decompress_from(&mut self, method: &str, n_chunks: usize, compressed_buffer: &Self) {
        self.inner
            .decompress_from(method, n_chunks, &compressed_buffer.inner);
    }

    pub fn data_ptr(&self) -> u64 {
        self.inner.data_ptr()
    }

    pub fn device_id(&self) -> usize {
        self.inner.device_id()
    }

    pub fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    pub fn num_elements_allocated(&self) -> usize {
        self.inner.num_elements_allocated()
    }

    pub fn dtype(&self) -> String {
        self.inner.dtype()
    }
}

#[pyclass(dict)]
pub struct BaguaCommBackendPy {
    inner: BaguaCommBackend,
}

#[pymethods]
impl BaguaCommBackendPy {
    #[new]
    pub fn new(schedule_channel_cap: usize, device_id: usize) -> Self {
        Self {
            inner: BaguaCommBackend::new(schedule_channel_cap, device_id),
        }
    }

    /// calling a second time will overwrite previous buckets
    pub fn register_ordered_buckets(
        &mut self,
        buckets: Vec<PyRef<BaguaBucketPy>>,
        py: Python,
    ) -> PyResult<()> {
        let mut buckets_inner = Vec::with_capacity(buckets.len());
        for b in buckets.iter() {
            buckets_inner.push(&b.inner)
        }
        py.allow_threads(|| {
            self.inner
                .register_ordered_buckets(buckets_inner.as_slice())
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    pub fn mark_communication_ready(
        &mut self,
        tensor: PyRef<BaguaTensorPy>,
        ready_cuda_event_ptr: u64,
        py: Python,
    ) -> PyResult<()> {
        let inner = &tensor.inner;
        py.allow_threads(|| {
            self.inner
                .mark_communication_ready(inner, ready_cuda_event_ptr)
        })
        .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }

    pub fn wait_pending_comm_ops(&self, py: Python) -> PyResult<usize> {
        py.allow_threads(|| self.inner.wait_pending_comm_ops())
            .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))
    }
}

#[pyclass(dict)]
pub struct BaguaBucketPy {
    inner: BaguaBucket,
}

#[pymethods]
impl BaguaBucketPy {
    #[new]
    pub fn new(name: &str, tensors: Vec<PyRef<BaguaTensorPy>>) -> PyResult<Self> {
        let mut tensors_inner = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            tensors_inner.push(&t.inner)
        }
        Ok(Self {
            inner: BaguaBucket::new(tensors_inner.as_slice(), name)
                .map_err(|e| PyRuntimeError::new_err(format!("{:?}", e)))?,
        })
    }

    pub fn tensors(&self) -> Vec<BaguaTensorPy> {
        self.inner
            .tensors()
            .into_iter()
            .map(|x| BaguaTensorPy { inner: x })
            .collect()
    }

    pub fn append_python_op(&mut self, op: &PyAny) -> PyResult<()> {
        assert!(op.is_callable(), "python op should be a callable");
        self.inner.append_python_op(op.into_py(op.py()));
        Ok(())
    }

    /// this function will use communicator_internode to communicate.
    /// if hierarchical = True, it will do hierarchical communicator, this requires intranode communicator on each node and inter node communicator on leader GPU. leader GPU will be the GPU whose communicator_intranode rank is 0
    #[args(average = "true", hierarchical = "false", scattergather = "false")]
    pub fn append_centralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicatorPy>,
        communicator_intranode: Option<&BaguaSingleCommunicatorPy>,
        hierarchical: bool,
        average: bool,
        scattergather: bool,
        compression: Option<String>,
    ) -> PyResult<()> {
        self.inner.append_centralized_synchronous_op(
            communicator_internode.map(|x| &x.inner),
            communicator_intranode.map(|x| &x.inner),
            hierarchical,
            average,
            scattergather,
            compression,
        );
        Ok(())
    }

    #[args(hierarchical = "false")]
    pub fn append_decentralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicatorPy>,
        communicator_intranode: Option<&BaguaSingleCommunicatorPy>,
        hierarchical: bool,
        peer_selection_mode: String,
        peer_weight: PyRef<BaguaTensorPy>,
    ) -> PyResult<()> {
        self.inner.append_decentralized_synchronous_op(
            communicator_internode.map(|x| &x.inner),
            communicator_intranode.map(|x| &x.inner),
            hierarchical,
            peer_selection_mode,
            (*peer_weight).inner.clone(),
        );
        Ok(())
    }

    #[args(hierarchical = "false")]
    pub fn append_low_precision_decentralized_synchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicatorPy>,
        communicator_intranode: Option<&BaguaSingleCommunicatorPy>,
        hierarchical: bool,
        peer_selection_mode: String,
        compression: String,
        weight: PyRef<BaguaTensorPy>,
        left_peer_weight: PyRef<BaguaTensorPy>,
        right_peer_weight: PyRef<BaguaTensorPy>,
    ) -> PyResult<()> {
        self.inner
            .append_low_precision_decentralized_synchronous_op(
                communicator_internode.map(|x| &x.inner),
                communicator_intranode.map(|x| &x.inner),
                hierarchical,
                peer_selection_mode,
                compression,
                (*weight).inner.clone(),
                (*left_peer_weight).inner.clone(),
                (*right_peer_weight).inner.clone(),
            );
        Ok(())
    }

    pub fn append_decentralized_asynchronous_op(
        &mut self,
        communicator_internode: Option<&BaguaSingleCommunicatorPy>,
        communicator_intranode: Option<&BaguaSingleCommunicatorPy>,
        peer_selection_mode: String,
        torch_stream: u64,
    ) -> DecentralizedFullPrecisionAsynchronousPy {
        DecentralizedFullPrecisionAsynchronousPy {
            inner: self.inner.append_decentralized_asynchronous_op(
                communicator_internode.map(|x| &x.inner),
                communicator_intranode.map(|x| &x.inner),
                peer_selection_mode,
                torch_stream,
            ),
        }
    }

    pub fn print_ops(&self) -> PyResult<()> {
        println!("{:?}", self.inner.inner.lock().comm_ops);
        Ok(())
    }

    pub fn clear_ops(&mut self) -> PyResult<()> {
        self.inner.inner.lock().comm_ops.clear();
        Ok(())
    }

    pub fn ready_for_comm(&self) -> bool {
        self.inner.ready_for_comm()
    }

    pub fn reset_comm_ready(&self) {
        self.inner.reset_comm_ready()
    }
}

#[pymodule]
fn bagua_core(_py: Python, m: &PyModule) -> PyResult<()> {
    if std::env::var("LOG_LEVEL").is_err() {
        std::env::set_var("LOG_LEVEL", "WARN");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_env("LOG_LEVEL"))
        .init();
    color_eyre::install().unwrap();

    // panic the whole process when thread panics
    let orig_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        // invoke the default handler and exit the process
        orig_hook(panic_info);
        std::process::exit(1);
    }));

    m.add_class::<BaguaCommBackendPy>()?;
    m.add_class::<BaguaTensorPy>()?;
    m.add_class::<BaguaBucketPy>()?;
    m.add_class::<BaguaSingleCommunicatorPy>()?;
    m.add_class::<DecentralizedFullPrecisionAsynchronousPy>()?;

    #[pyfn(m, "show_version")]
    fn show_version(_py: Python) {
        bagua_core_internal::show_version();
    }

    std::thread::spawn(move || {
        tracing::info!("deadlock detection thread started");
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            let deadlocks = parking_lot::deadlock::check_deadlock();
            if deadlocks.is_empty() {
                continue;
            }

            tracing::error!("{} deadlocks detected", deadlocks.len());
            for (i, threads) in deadlocks.iter().enumerate() {
                tracing::error!("Deadlock #{}", i);
                for t in threads {
                    tracing::error!("Thread Id {:#?}", t.thread_id());
                    tracing::error!("{:#?}", t.backtrace());
                }
            }
        }
    });

    Ok(())
}
