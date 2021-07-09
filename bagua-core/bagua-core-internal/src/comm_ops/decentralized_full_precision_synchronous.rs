use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensorRaw, RawBaguaTensor};
use crate::events::BaguaEventChannel;
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::{BaguaCommOpChannels, BaguaScheduledCommOp};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum PeerSelectionMode {
    All,
    ShiftOne,
    Ring,
}

#[derive(Debug)]
pub struct DecentralizedFullPrecisionSynchronous {
    pub communicator: BaguaCommunicator,
    pub peer_selection_mode: PeerSelectionMode,
    pub step: Mutex<usize>,
    pub communication_interval: usize,
}

impl CommOpTrait for DecentralizedFullPrecisionSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket_guard = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();

        let mut communication_tensor = match &self.communicator {
            BaguaCommunicator::SingleCommunicator(_) => {
                bucket_guard.get_communication_tensor(stream_ptr, false, false)
            }
            BaguaCommunicator::HierarchicalCommunicator(x) => match x {
                BaguaHierarchicalCommunicator::Leader(_) => {
                    bucket_guard.get_communication_tensor(stream_ptr, true, true)
                }
                BaguaHierarchicalCommunicator::Worker(_) => {
                    bucket_guard.get_communication_tensor(stream_ptr, false, false)
                }
            },
        };

        let t = &communication_tensor;
        let peer_tensor_buffer = CUDA_DEVICE_MEMORY_POOL[t.raw.device_id]
            .try_pull(t.raw.num_elem_allocated * t.raw.dtype.bytes())
            .expect("cannot allocate gpu memory");
        let mut peer_tensor = BaguaTensorRaw {
            ptr: peer_tensor_buffer.ptr,
            num_elem_allocated: t.raw.num_elem_allocated,
            dtype: t.raw.dtype,
            num_elem: t.raw.num_elem,
            device_id: t.raw.device_id,
            pool_allocations: vec![Arc::new(peer_tensor_buffer)],
        };

        let peer_mode = &self.peer_selection_mode;
        let comm_interval = &self.communication_interval;
        let step = { *self.step.lock() };

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {
                match peer_mode {
                    PeerSelectionMode::All => {
                        {
                            if step % comm_interval == 0 {
                                peer_tensor.clone_from(&t.raw, c.stream_ptr);
                                let _guard = NCCLGroupGuard::new();
                                c.allreduce(&mut peer_tensor, BaguaReductionOp::SUM);
                                peer_tensor.divide_inplace(stream_ptr, c.nranks as f32);
                            }
                        }
                    }
                    PeerSelectionMode::ShiftOne => {
                        if step % comm_interval == 0 {
                            assert_eq!(
                                c.nranks % 2,
                                0,
                                "you cannot use decentralized algorithm with average_all off when there are odd number of ranks, current n_ranks {}",
                                c.nranks
                            );
                            let comm_step = step / comm_interval;
                            let peer_rank = if c.rank < c.nranks / 2 {
                                ((comm_step + c.rank) % ((c.nranks + 1) / 2)) + (c.nranks / 2)
                            } else {
                                (c.rank - (c.nranks / 2) - comm_step).rem_euclid(c.nranks / 2)
                            } as i32;
                            tracing::debug!("rank {} peer_rank {}", c.rank, peer_rank);
                            {
                                let _guard = NCCLGroupGuard::new();
                                c.send(&t.raw, peer_rank);
                                c.recv(&mut peer_tensor, peer_rank);
                            }
                            peer_tensor.average_inplace(&t.raw, c.stream_ptr);
                        }
                    },
                    PeerSelectionMode::Ring => {
                        unimplemented!()
                    }
                }
            },
        );

        if step % comm_interval == 0 {
            // TODO: move this to .then() python API instead of hard code this in op
            let post_backward_comm_op = BaguaScheduledCommOp {
                name: format!("post backward comm op for bucket {}", bucket.name),
                bucket: bucket.clone(),
                ops: vec![Arc::new(DecentralizedFullPrecisionSynchronousPostStep {
                    communicator: self.communicator.clone(),
                    result_weight: peer_tensor,
                })],
                event_channel: BaguaEventChannel::new("decentralized_post_backward"),
            };

            comm_op_channels
                .not_waited_post_backward_events_sender
                .send(post_backward_comm_op.event_channel.clone())
                .expect("cannot send post backward event");
            comm_op_channels
                .post_backward_channel_sender
                .send(post_backward_comm_op)
                .expect("cannot send post backward op");
        }

        *self.step.lock() += 1;
    }
}

#[derive(Debug)]
pub struct DecentralizedFullPrecisionSynchronousPostStep {
    pub communicator: BaguaCommunicator,
    pub result_weight: BaguaTensorRaw,
}

impl CommOpTrait for DecentralizedFullPrecisionSynchronousPostStep {
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
            false,
            false,
            true,
            &mut |c, t| {
                t.raw.clone_from(&self.result_weight, c.stream_ptr);
            },
        );
    }
}
