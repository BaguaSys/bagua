use crate::comm_ops::decentralized_full_precision_synchronous::PeerSelectionMode;
use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor};
use crate::events::BaguaEventChannel;
use crate::resource_pool::{CUDA_DEVICE_MEMORY_POOL, CUDA_EVENT_POOL};
use crate::{BaguaCommOpChannels, BaguaCoreError};
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug)]
pub struct DecentralizedFullPrecisionAsynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub torch_stream: u64,
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
             tracing::debug!("async model average start");

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

             // use default stream to copy weights
             temp_tensor.clone_from(&t.raw, torch_stream as u64);

             let src_ready_event = CUDA_EVENT_POOL.take().event;

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

             let comm_ready_event = CUDA_EVENT_POOL.take().event;

             unsafe {
                 cpp::cpp!([
                     comm_ready_event as "cudaEvent_t",
                     comm_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaEventRecord(comm_ready_event, comm_stream));
                     CUDACHECK(cudaEventSynchronize(comm_ready_event));
                 });
             }

             if c.check_abort() {
                 tracing::debug!("async model average on process {} early stopped due to communicator abortion", c.rank);
                 return
             }


             // do we need to wait default stream?
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

             t.raw.async_model_average(&reduced_tensor, &temp_tensor, c.nranks as f32, comm_stream);

             unsafe {
                 cpp::cpp!([comm_stream as "cudaStream_t"]
                 {
                     CUDACHECK(cudaStreamSynchronize(comm_stream));
                 });
             }

             tracing::debug!("async model average update cost: {:?}", start_time.elapsed());

        });
    }
}
