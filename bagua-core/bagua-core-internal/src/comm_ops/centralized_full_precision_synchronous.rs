use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedFullPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    /// whether divide world_size after allreduce sum op
    pub average: bool,
    pub scattergather: bool,
}

impl CommOpTrait for CentralizedFullPrecisionSynchronous {
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
                tracing::debug!("internode communication started");
                let temp_buf = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
                    .try_pull(t.raw.num_elem_allocated * t.raw.dtype.bytes())
                    .expect("cannot allocate cuda memory");
                let mut temp_tensor = BaguaTensorRaw {
                    ptr: temp_buf.ptr,
                    num_elem_allocated: t.raw.num_elem_allocated,
                    dtype: t.raw.dtype.clone(),
                    num_elem: t.raw.num_elem,
                    device_id: t.raw.device_id,
                    pool_allocations: vec![Arc::new(temp_buf)],
                };
                if self.scattergather {
                    tracing::debug!("start alltoall");
                    c.alltoall(&t.raw, &mut temp_tensor);
                    tracing::debug!("start reduce_sum");
                    if self.average {
                        temp_tensor.reduce_mean_inplace(c.nranks, c.rank, c.stream_ptr);
                    } else {
                        temp_tensor.reduce_sum_inplace(c.nranks, c.rank, c.stream_ptr);
                    }
                    tracing::debug!("start allgather");
                    c.allgather(&temp_tensor, &mut t.raw);
                    tracing::debug!("internode communication done")
                } else {
                    tracing::debug!("start allreduce");
                    c.allreduce(&mut t.raw, BaguaReductionOp::SUM);
                    tracing::debug!("internode communication done");
                    if self.average {
                        t.raw.divide_inplace(stream_ptr, c.nranks as f32);
                    }
                }
            },
        );
    }
}
