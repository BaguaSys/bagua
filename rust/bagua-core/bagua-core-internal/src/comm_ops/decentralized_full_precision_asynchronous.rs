use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::cuda_utils::{cuda_memcpy_device_to_host_sync, cuda_memcpy_host_to_device_sync};
use crate::datatypes::{
    BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorDtype, BaguaTensorRaw, RawBaguaTensor,
};
use crate::events::BaguaEventChannel;
use crate::resource_pool::{CUDA_DEVICE_MEMORY_POOL, CUDA_EVENT_POOL};
use crate::{BaguaCommOpChannels, BaguaCoreError};
use parking_lot::{lock_api::RawMutex as _, Mutex, RawMutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub struct DecentralizedFullPrecisionAsynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub torch_stream: u64,
    pub weight_mutex: Arc<Mutex<bool>>,
    pub aborted: Arc<AtomicBool>,
    pub all_aborted: Arc<AtomicBool>,
}

impl CommOpTrait for DecentralizedFullPrecisionAsynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket_guard = bucket.inner.lock();

        let comm_stream = self.communicator.stream_ptr();

        let mut communication_tensor = match &self.communicator {
            BaguaCommunicator::SingleCommunicator(_) => {
                bucket_guard.get_communication_tensor(comm_stream, false, false)
            }
            BaguaCommunicator::HierarchicalCommunicator(x) => {
                panic!("asynchronous op only accepts non-hierarchical communicator");
            }
        };

        let peer_mode = &self.peer_selection_mode;

        let torch_stream = self.torch_stream;

        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            false,
            false,
            &mut |c, t| {
                let start_time = std::time::Instant::now();
                tracing::debug!("#{} async model average start", c.rank);

                let state_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
                    .try_pull(1)
                    .expect("cannot allocate cuda memory");

                let mut state_tensor = BaguaTensorRaw {
                    ptr: state_buf.ptr,
                    num_elem_allocated: 1,
                    dtype: BaguaTensorDtype::U8,
                    num_elem: 1,
                    device_id: t.raw.device_id(),
                    pool_allocations: vec![Arc::new(state_buf)],
                };

                let mut aborted = { self.aborted.load(Ordering::Relaxed) } as u8;
                let aborted_ptr: *mut u8 = &mut aborted;
                unsafe {
                    cuda_memcpy_host_to_device_sync(state_tensor.data_ptr(), aborted_ptr as u64, 1);
                }

                let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
                    .try_pull(t.raw.num_elements_allocated() * t.raw.dtype().bytes())
                    .expect("cannot allocate cuda memory");

                let mut temp_tensor = BaguaTensorRaw {
                    ptr: temp_buf.ptr,
                    num_elem_allocated: t.raw.num_elements_allocated(),
                    dtype: t.raw.dtype().clone(),
                    num_elem: t.raw.num_elements(),
                    device_id: t.raw.device_id(),
                    pool_allocations: vec![Arc::new(temp_buf)],
                };

                let reduced_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id()]
                    .try_pull(t.raw.num_elements_allocated() * t.raw.dtype().bytes())
                    .expect("cannot allocate cuda memory");

                let mut reduced_tensor = BaguaTensorRaw {
                    ptr: reduced_buf.ptr,
                    num_elem_allocated: t.raw.num_elements_allocated(),
                    dtype: t.raw.dtype().clone(),
                    num_elem: t.raw.num_elements(),
                    device_id: t.raw.device_id(),
                    pool_allocations: vec![Arc::new(reduced_buf)],
                };

                let ready_event = CUDA_EVENT_POOL.take().event;
                let src_ready_event = CUDA_EVENT_POOL.take().event;

                // use default stream to copy weights
                temp_tensor.clone_from(&t.raw, torch_stream as u64);

                c.allreduce_inplace(&mut state_tensor, BaguaReductionOp::MIN);

                unsafe {
                    cpp::cpp!([comm_stream as "cudaStream_t"]
                    {
                        CUDACHECK(cudaStreamSynchronize(comm_stream));
                    });
                }

                unsafe {
                    cuda_memcpy_device_to_host_sync(aborted_ptr as u64, state_tensor.data_ptr(), 1);
                }

                if aborted > 0 {
                    tracing::debug!("#{} async model average aborted", c.rank);

                    self.all_aborted.store(true, Ordering::Relaxed);
                    return;
                }

                unsafe {
                    cpp::cpp!([
                        src_ready_event as "cudaEvent_t",
                        comm_stream as "cudaStream_t",
                        torch_stream as "cudaStream_t"]
                    {
                        CUDACHECK(cudaEventRecord(src_ready_event, torch_stream));
                        CUDACHECK(cudaStreamWaitEvent(comm_stream, src_ready_event , 0));
                    });
                }

                match peer_mode {
                    PeerSelectionMode::All => {
                        c.allreduce(&temp_tensor, &mut reduced_tensor, BaguaReductionOp::SUM);
                    }
                    PeerSelectionMode::Ring => {
                        unimplemented!()
                    }
                    PeerSelectionMode::ShiftOne => {
                        unimplemented!()
                    }
                };

                {
                    unsafe {
                        cpp::cpp!([
                            ready_event as "cudaEvent_t",
                            comm_stream as "cudaStream_t"]
                        {
                            CUDACHECK(cudaEventRecord(ready_event, comm_stream));
                            CUDACHECK(cudaEventSynchronize(ready_event));
                        });
                    }

                    self.lock_weight();
                    t.raw.async_model_average(
                        &reduced_tensor,
                        &temp_tensor,
                        c.nranks as f32,
                        torch_stream,
                    );

                    unsafe {
                        cpp::cpp!([
                            ready_event as "cudaEvent_t",
                            torch_stream as "cudaStream_t"]
                        {
                            CUDACHECK(cudaEventRecord(ready_event, torch_stream));
                            CUDACHECK(cudaEventSynchronize(ready_event));
                        });
                    }
                    self.unlock_weight();
                }

                tracing::debug!(
                    "#{} async model average update cost: {:?}",
                    c.rank,
                    start_time.elapsed()
                );
            },
        );
    }
}

impl DecentralizedFullPrecisionAsynchronous {
    pub fn lock_weight(&self) {
        let raw_mutex = unsafe { self.weight_mutex.raw() };
        raw_mutex.lock();
    }

    pub fn unlock_weight(&self) {
        unsafe {
            let raw_mutex = self.weight_mutex.raw();
            raw_mutex.unlock();
        };
    }

    pub fn abort(&self) {
        self.aborted.store(true, Ordering::Relaxed);
    }

    pub fn reset(&self) {
        self.aborted.store(false, Ordering::Relaxed);
        self.all_aborted.store(false, Ordering::Relaxed);
    }

    pub fn is_all_aborted(&self) -> bool {
        self.all_aborted.load(Ordering::Relaxed)
    }
}
