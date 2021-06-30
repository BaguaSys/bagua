use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaTensorRaw, RawBaguaTensor, TensorCompressionMethod};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedLowPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    /// whether divide world_size after allreduce sum op
    pub average: bool,
    pub compression_method: TensorCompressionMethod,
}

impl CommOpTrait for CentralizedLowPrecisionSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = bucket.get_communication_tensor(stream_ptr, false, false);
        self.communicator.execute_communication(
            &mut communication_tensor,
            self.average,
            true,
            true,
            &mut |c, t| {
                tracing::debug!("start compress");
                let compressed_tensor = t
                    .raw
                    .compress(&self.compression_method, c.nranks, c.stream_ptr, -1)
                    .expect("cannot compress tensor");
                let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
                    .try_pull(
                        compressed_tensor.num_elements_allocated()
                            * compressed_tensor.dtype().bytes(),
                    )
                    .expect("cannot allocate cuda memory");
                let mut temp_tensor = BaguaTensorRaw {
                    ptr: temp_buf.ptr,
                    num_elem_allocated: compressed_tensor.num_elements_allocated(),
                    dtype: compressed_tensor.dtype().clone(),
                    num_elem: compressed_tensor.num_elements(),
                    device_id: compressed_tensor.device_id(),
                    pool_allocations: vec![Arc::new(temp_buf)],
                };
                tracing::debug!("start alltoall");
                c.alltoall(compressed_tensor.as_ref(), &mut temp_tensor);
                tracing::debug!("start decompress");
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &temp_tensor,
                    c.stream_ptr,
                );
                tracing::debug!("start reduce_sum");
                if self.average {
                    t.raw.reduce_mean_inplace(c.nranks, c.rank, c.stream_ptr);
                } else {
                    t.raw.reduce_sum_inplace(c.nranks, c.rank, c.stream_ptr);
                }
                tracing::debug!("start compress");
                let compressed_tensor = t
                    .raw
                    .compress(
                        &self.compression_method,
                        c.nranks,
                        c.stream_ptr,
                        c.rank as _,
                    )
                    .expect("cannot compress tensor");
                tracing::debug!("start allgather");
                c.allgather(compressed_tensor.as_ref(), &mut temp_tensor);
                tracing::debug!("start decompress");
                t.raw.decompress_from(
                    &self.compression_method,
                    c.nranks,
                    &temp_tensor,
                    c.stream_ptr,
                );
                tracing::debug!("internode communication done");
            },
        );
    }
}
