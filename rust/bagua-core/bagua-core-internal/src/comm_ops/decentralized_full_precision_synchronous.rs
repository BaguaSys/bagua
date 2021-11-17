use crate::comm_ops::CommOpTrait;
use crate::communicators::{BaguaCommunicator, BaguaHierarchicalCommunicator, NCCLGroupGuard};
use crate::datatypes::{
    BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor,
};
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
    pub peer_weight: BaguaTensor,
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

        let peer_mode = &self.peer_selection_mode;

        let mut peer_guard = self.peer_weight.inner.write();
        let mut peer_tensor = peer_guard.raw.as_mut();
        let step = { *self.step.lock() } as i64;

        self.communicator.execute_communication(
            &mut communication_tensor,
            true,
            true,
            false,
            &mut |c, t| {
                match peer_mode {
                    PeerSelectionMode::All => {
                        {
                            peer_tensor.clone_from(&t.raw, c.stream_ptr);
                            let _guard = NCCLGroupGuard::new();
                            c.allreduce_inplace(peer_tensor, BaguaReductionOp::AVG);
                        }
                    }
                    PeerSelectionMode::ShiftOne => {
                        assert_eq!(
                            c.nranks % 2,
                            0,
                            "You cannot use decentralized algorithm with average_all off when there are odd number of ranks, current n_ranks {}. \
                            Note that by default hierarchical communication is enabled, where n_ranks will become the number of machines instead of number of GPUs. \
                            See https://bagua.readthedocs.io/en/latest/autoapi/bagua/torch_api/algorithms/decentralized/index.html for details.",
                            c.nranks
                        );
                        let rank = c.rank as i64;
                        let nranks = c.nranks as i64;
                        let peer_rank = if c.rank < c.nranks / 2 {
                            ((step + rank) % ((nranks + 1) / 2)) + (nranks / 2)
                        } else {
                            (rank - (nranks / 2) - step).rem_euclid(nranks / 2)
                        } as i32;
                        tracing::debug!("rank {} peer_rank {}", c.rank, peer_rank);
                        {
                            let _guard = NCCLGroupGuard::new();
                            c.send(&t.raw, peer_rank);
                            c.recv(peer_tensor, peer_rank);
                        }
                        peer_tensor.average_inplace(&t.raw, c.stream_ptr);
                    },
                    PeerSelectionMode::Ring => {
                        unimplemented!()
                    },
                }
            },
        );

        *self.step.lock() += 1;
    }
}

impl DecentralizedFullPrecisionSynchronous {
    pub fn copy_back_peer_weight(&self, bucket: Arc<BaguaBucket>) {
        let bucket_guard = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();

        let mut communication_tensor =
            bucket_guard.get_communication_tensor(stream_ptr, false, false);

        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            false,
            true,
            &mut |c, t| {
                t.raw
                    .clone_from(self.peer_weight.inner.read().raw.as_ref(), c.stream_ptr);
            },
        );
    }
}
