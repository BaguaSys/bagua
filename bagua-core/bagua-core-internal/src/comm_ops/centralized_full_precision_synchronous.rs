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
                if self.scattergather {
                    tracing::debug!("start alltoall");
                    c.alltoall_inplace(&mut t.raw);
                    tracing::debug!("start reduce_sum");
                    if self.average {
                        t.raw.reduce_mean_inplace(c.nranks, c.rank, c.stream_ptr);
                    } else {
                        t.raw.reduce_sum_inplace(c.nranks, c.rank, c.stream_ptr);
                    }
                    tracing::debug!("start allgather");
                    c.allgather_inplace(&mut t.raw);
                    tracing::debug!("internode communication done")
                } else {
                    tracing::debug!("start allreduce");
                    if self.average {
                        c.allreduce_inplace(&mut t.raw, BaguaReductionOp::AVG);
                    } else {
                        c.allreduce_inplace(&mut t.raw, BaguaReductionOp::SUM);
                    }
                    tracing::debug!("internode communication done");
                }
            },
        );
    }
}
