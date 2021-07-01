use crate::datatypes::{
    BaguaCommunicationTensor, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor,
};
use crate::BaguaCoreError;
use itertools::Itertools;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct BaguaCommunicatorInner {
    pub stream_ptr: u64,
    pub comm_ptr: u64,
    pub rank: usize,
    pub nranks: usize,
    pub device_id: usize,
}

#[derive(Clone, Debug)]
pub struct BaguaSingleCommunicator {
    pub inner: Arc<BaguaCommunicatorInner>,
}

impl BaguaSingleCommunicator {
    pub fn new(
        rank: usize,
        nranks: usize,
        device_id: usize,
        stream_ptr: u64,
        nccl_unique_id_str: &str,
    ) -> Self {
        unsafe {
            cpp::cpp!([device_id as "size_t"]
            { CUDACHECK(cudaSetDevice(device_id)); });
        }

        tracing::debug!("creating communicator, nccl_unique_id {}, rank {}, nranks {}, device_id {}, stream_ptr {}", nccl_unique_id_str, rank, nranks, device_id, stream_ptr);
        let bytes = base64::decode(nccl_unique_id_str).unwrap();
        let al_comm_ptr = unsafe {
            let bytes_ptr = bytes.as_ptr();
            cpp::cpp!([rank as "size_t", nranks as "size_t", bytes_ptr as "char *", stream_ptr as "cudaStream_t"] -> u64 as "Al::NCCLCommunicator *" {
                ncclUniqueId id;
                memcpy(id.internal, bytes_ptr, NCCL_UNIQUE_ID_BYTES);
                Al::NCCLCommunicator* nccl_comm = new Al::NCCLCommunicator(rank, nranks, id, stream_ptr);
                return nccl_comm;
            })
        };
        tracing::debug!("al communicator initialized at {}", al_comm_ptr,);

        Self {
            inner: Arc::new(BaguaCommunicatorInner {
                stream_ptr,
                comm_ptr: al_comm_ptr as _,
                rank,
                nranks,
                device_id,
            }),
        }
    }

    pub fn nranks(&self) -> usize {
        self.inner.nranks
    }

    pub fn rank(&self) -> usize {
        self.inner.rank
    }

    pub fn device_id(&self) -> usize {
        self.inner.device_id
    }

    pub fn allreduce(&self, tensor: &mut BaguaTensor, op: BaguaReductionOp) {
        self.inner.allreduce(tensor.inner.write().raw.as_mut(), op);
    }

    pub fn broadcast(&self, tensor: &mut BaguaTensor, root_rank: i32) {
        self.inner
            .broadcast(tensor.inner.write().raw.as_mut(), root_rank);
    }

    pub fn reduce(&self, tensor: &mut BaguaTensor, root_rank: i32, op: BaguaReductionOp) {
        self.inner
            .reduce(tensor.inner.write().raw.as_mut(), root_rank, op);
    }

    pub fn send(&self, tensor: &mut BaguaTensor, peer_rank: i32) {
        self.inner
            .send(tensor.inner.write().raw.as_mut(), peer_rank);
    }

    pub fn recv(&self, tensor: &mut BaguaTensor, peer_rank: i32) {
        self.inner
            .recv(tensor.inner.write().raw.as_mut(), peer_rank);
    }

    pub fn alltoall(&self, send_tensor: &mut BaguaTensor, recv_tensor: &mut BaguaTensor) {
        self.inner.alltoall(
            send_tensor.inner.write().raw.as_mut(),
            recv_tensor.inner.write().raw.as_mut(),
        );
    }

    pub fn alltoall_v(
        &self,
        send_tensor: &mut BaguaTensor,
        send_counts: &[usize],
        send_displs: &[usize],
        recv_tensor: &mut BaguaTensor,
        recv_counts: &[usize],
        recv_displs: &[usize],
    ) {
        self.inner.alltoall_v(
            send_tensor.inner.write().raw.as_mut(),
            send_counts,
            send_displs,
            recv_tensor.inner.write().raw.as_mut(),
            recv_counts,
            recv_displs,
        );
    }

    pub fn allgather(&self, send_tensor: &mut BaguaTensor, recv_tensor: &mut BaguaTensor) {
        self.inner.allgather(
            send_tensor.inner.write().raw.as_mut(),
            recv_tensor.inner.write().raw.as_mut(),
        );
    }

    pub fn barrier(&self) {
        self.inner.barrier();
    }

    pub fn generate_nccl_unique_id_str() -> String {
        let unique_id = vec![0_i8; 128];
        let unique_id_ptr = unique_id.as_ptr();
        unsafe {
            cpp::cpp!([unique_id_ptr as "char *"]
            {
              ncclUniqueId id;
              ncclGetUniqueId(&id);
              memcpy(unique_id_ptr, id.internal, NCCL_UNIQUE_ID_BYTES);
            });
        }
        let id_str = base64::encode(unique_id.iter().map(|x| *x as u8).collect_vec());
        tracing::debug!("nccl_unique_id generated: {}", id_str);
        id_str
    }
}

#[derive(Clone, Debug)]
pub struct BaguaHierarchicalCommunicatorLeader {
    pub internode: BaguaSingleCommunicator,
    pub intranode: BaguaSingleCommunicator,
}

impl BaguaHierarchicalCommunicatorLeader {
    pub fn new(internode: BaguaSingleCommunicator, intranode: BaguaSingleCommunicator) -> Self {
        {
            assert_eq!(intranode.inner.stream_ptr, internode.inner.stream_ptr);
        }
        {
            assert_eq!(intranode.inner.device_id, internode.inner.device_id);
        }

        Self {
            internode,
            intranode,
        }
    }

    pub fn hierarchical_pre(
        &self,
        communication_tensor: &mut BaguaCommunicationTensor,
        average: bool,
    ) {
        let intranode_communicator = self.intranode.inner.clone();
        let stream_ptr = intranode_communicator.stream_ptr;
        assert_eq!(communication_tensor.stream_ptr, stream_ptr);
        tracing::debug!("reduce start");
        intranode_communicator.reduce(&mut communication_tensor.raw, 0, BaguaReductionOp::SUM);
        tracing::debug!("reduce done");
        if average {
            communication_tensor
                .raw
                .divide_inplace(stream_ptr, (intranode_communicator.nranks) as f32);
        }
    }

    pub fn hierarchical_post(&self, communication_tensor: &mut BaguaCommunicationTensor) {
        let intranode_communicator = self.intranode.inner.clone();
        let stream_ptr = intranode_communicator.stream_ptr;
        assert_eq!(communication_tensor.stream_ptr, stream_ptr);
        tracing::debug!("broadcast start");
        intranode_communicator.broadcast(&mut communication_tensor.raw, 0);
        tracing::debug!("broadcast done");
    }
}

#[derive(Clone, Debug)]
pub struct BaguaHierarchicalCommunicatorWorker {
    pub intranode: BaguaSingleCommunicator,
}

impl BaguaHierarchicalCommunicatorWorker {
    pub fn hierarchical_worker_pre(&self, communication_tensor: &mut BaguaCommunicationTensor) {
        let intranode_communicator = self.intranode.inner.clone();
        intranode_communicator.reduce(&mut communication_tensor.raw, 0, BaguaReductionOp::SUM);
    }

    pub fn hierarchical_worker_post(&self, communication_tensor: &mut BaguaCommunicationTensor) {
        let intranode_communicator = self.intranode.inner.clone();
        intranode_communicator.broadcast(&mut communication_tensor.raw, 0);
    }
}

#[derive(Clone, Debug)]
pub enum BaguaHierarchicalCommunicator {
    Leader(BaguaHierarchicalCommunicatorLeader),
    Worker(BaguaHierarchicalCommunicatorWorker),
}

#[derive(Clone, Debug)]
pub enum BaguaCommunicator {
    SingleCommunicator(BaguaSingleCommunicator),
    HierarchicalCommunicator(BaguaHierarchicalCommunicator),
}

impl BaguaCommunicator {
    pub fn new(
        communicator_internode: Option<&BaguaSingleCommunicator>,
        communicator_intranode: Option<&BaguaSingleCommunicator>,
        hierarchical: bool,
    ) -> Result<Self, BaguaCoreError> {
        match hierarchical {
            false => Ok(BaguaCommunicator::SingleCommunicator(
                communicator_internode
                    .expect("inter node communicator must be given in non-hierarchical mode")
                    .clone(),
            )),
            true => {
                let intranode_rank = communicator_intranode.as_ref().unwrap().rank();
                if intranode_rank == 0 {
                    let intra = communicator_intranode.expect("intra node communicator must be given in worker GPU in hierarchical mode").clone();
                    let inter = communicator_internode.unwrap().clone();
                    {
                        if intra.inner.stream_ptr != inter.inner.stream_ptr {
                            return Err(BaguaCoreError::CommunicatorError("intra node communicator should use the same stream as the inter node communicator".into()));
                        }
                    }
                    Ok(BaguaCommunicator::HierarchicalCommunicator(
                        BaguaHierarchicalCommunicator::Leader(
                            BaguaHierarchicalCommunicatorLeader::new(inter, intra),
                        ),
                    ))
                } else {
                    Ok(BaguaCommunicator::HierarchicalCommunicator(BaguaHierarchicalCommunicator::Worker(BaguaHierarchicalCommunicatorWorker {
                        intranode: communicator_intranode.expect("intra node communicator must be given in worker GPU in hierarchical mode").clone()
                    })))
                }
            }
        }
    }

    pub fn stream_ptr(&self) -> u64 {
        match self {
            BaguaCommunicator::SingleCommunicator(x) => x.inner.stream_ptr,
            BaguaCommunicator::HierarchicalCommunicator(x) => match x {
                BaguaHierarchicalCommunicator::Leader(x) => x.internode.inner.stream_ptr,
                BaguaHierarchicalCommunicator::Worker(x) => x.intranode.inner.stream_ptr,
            },
        }
    }

    pub fn execute_communication(
        &self,
        tensor: &mut BaguaCommunicationTensor,
        intranode_average: bool,
        hierarchical_pre: bool,
        hierarchical_post: bool,
        communication_hook: &mut dyn FnMut(
            &BaguaCommunicatorInner,
            &mut BaguaCommunicationTensor,
        ) -> (),
    ) {
        match &self {
            BaguaCommunicator::SingleCommunicator(communicator) => {
                let communicator = communicator.inner.clone();
                communication_hook(&communicator, tensor);
            }
            BaguaCommunicator::HierarchicalCommunicator(communicator) => match communicator {
                BaguaHierarchicalCommunicator::Leader(communicator) => {
                    let internode_communicator = communicator.internode.inner.clone();
                    if hierarchical_pre {
                        communicator.hierarchical_pre(tensor, intranode_average);
                    }
                    communication_hook(&internode_communicator, tensor);
                    if hierarchical_post {
                        communicator.hierarchical_post(tensor);
                    }
                }
                BaguaHierarchicalCommunicator::Worker(communicator) => {
                    if hierarchical_pre {
                        communicator.hierarchical_worker_pre(tensor);
                    }
                    if hierarchical_post {
                        communicator.hierarchical_worker_post(tensor);
                    }
                }
            },
        }
    }
}

pub struct NCCLGroupGuard {}

impl NCCLGroupGuard {
    pub fn new() -> Self {
        unsafe {
            cpp::cpp!([]
            {
                NCCLCHECK(ncclGroupStart());
            });
        }
        NCCLGroupGuard {}
    }
}

impl Drop for NCCLGroupGuard {
    fn drop(&mut self) {
        unsafe {
            cpp::cpp!([]
            {
                NCCLCHECK(ncclGroupEnd());
            });
        }
    }
}

impl BaguaCommunicatorInner {
    pub fn broadcast(&self, tensor: &mut dyn RawBaguaTensor, root_rank: i32) {
        let communicator_ptr = self.comm_ptr;
        let tensor_ptr = tensor.data_ptr();
        let total_num_elem = tensor.num_elements_allocated();
        let nccl_tensor_type = tensor.dtype().to_nccl_datatype();

        unsafe {
            cpp::cpp!([tensor_ptr as "void *", root_rank as "int", total_num_elem as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Bcast<Al::NCCLBackend>(static_cast<float*>(tensor_ptr), total_num_elem, root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Bcast<Al::NCCLBackend>(static_cast<__half*>(tensor_ptr), total_num_elem, root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Bcast<Al::NCCLBackend>(static_cast<unsigned char*>(tensor_ptr), total_num_elem, root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Bcast<Al::NCCLBackend>(static_cast<long long int*>(tensor_ptr), total_num_elem, root_rank, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn reduce(&self, tensor: &mut dyn RawBaguaTensor, root_rank: i32, op: BaguaReductionOp) {
        let communicator_ptr = self.comm_ptr;
        let tensor_ptr = tensor.data_ptr();
        let total_num_elem = tensor.num_elements_allocated();
        let nccl_tensor_type = tensor.dtype().to_nccl_datatype();

        unsafe {
            cpp::cpp!([tensor_ptr as "void *", root_rank as "int", total_num_elem as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t", op as "uint8_t"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Reduce<Al::NCCLBackend>(static_cast<float*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Reduce<Al::NCCLBackend>(static_cast<__half*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Reduce<Al::NCCLBackend>(static_cast<unsigned char*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), root_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Reduce<Al::NCCLBackend>(static_cast<long long int*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), root_rank, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn alltoall(&self, send_tensor: &dyn RawBaguaTensor, recv_tensor: &mut dyn RawBaguaTensor) {
        let communicator_ptr = self.comm_ptr;
        // TODO: also check recv buf?
        assert_eq!(
            send_tensor.num_elements_allocated() % self.nranks,
            0,
            "tensors must be aligned before using allscatter"
        );
        let send_chunk_size = send_tensor.num_elements_allocated() / self.nranks;
        let nccl_tensor_type = send_tensor.dtype().to_nccl_datatype();

        let send_buf_ptr = send_tensor.data_ptr();
        let recv_buf_ptr = recv_tensor.data_ptr();

        unsafe {
            cpp::cpp!([recv_buf_ptr as "void *", send_buf_ptr as "void *", send_chunk_size as "size_t", communicator_ptr as "Al::NCCLCommunicator *",  nccl_tensor_type as "ncclDataType_t"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Alltoall<Al::NCCLBackend>(static_cast<float*>(send_buf_ptr), static_cast<float*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Alltoall<Al::NCCLBackend>(static_cast<__half*>(send_buf_ptr), static_cast<__half*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Alltoall<Al::NCCLBackend>(static_cast<unsigned char*>(send_buf_ptr), static_cast<unsigned char*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Alltoall<Al::NCCLBackend>(static_cast<long long int*>(send_buf_ptr), static_cast<long long int*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn alltoall_v(
        &self,
        send_tensor: &dyn RawBaguaTensor,
        send_counts: &[usize],
        send_displs: &[usize],
        recv_tensor: &mut dyn RawBaguaTensor,
        recv_counts: &[usize],
        recv_displs: &[usize],
    ) {
        let communicator_ptr = self.comm_ptr;
        let nranks = self.nranks;
        let nccl_tensor_type = send_tensor.dtype().to_nccl_datatype();

        let send_buf_ptr = send_tensor.data_ptr();
        let send_counts_ptr = send_counts.as_ptr();
        let send_displs_ptr = send_displs.as_ptr();
        let recv_buf_ptr = recv_tensor.data_ptr();
        let recv_counts_ptr = recv_counts.as_ptr();
        let recv_displs_ptr = recv_displs.as_ptr();

        unsafe {
            cpp::cpp!([
                send_buf_ptr as "void *", send_counts_ptr as "size_t *", send_displs_ptr as "size_t *",
                recv_buf_ptr as "void *", recv_counts_ptr as "size_t *", recv_displs_ptr as "size_t *",
                nranks as "size_t", communicator_ptr as "Al::NCCLCommunicator *",  nccl_tensor_type as "ncclDataType_t"]
            {
                std::vector<size_t> send_counts(send_counts_ptr, send_counts_ptr + nranks);
                std::vector<size_t> send_displs(send_displs_ptr, send_displs_ptr + nranks);
                std::vector<size_t> recv_counts(recv_counts_ptr, recv_counts_ptr + nranks);
                std::vector<size_t> recv_displs(recv_displs_ptr, recv_displs_ptr + nranks);
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Alltoallv<Al::NCCLBackend>(static_cast<const float*>(send_buf_ptr), send_counts, send_displs, static_cast<float*>(recv_buf_ptr), recv_counts, recv_displs, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Alltoallv<Al::NCCLBackend>(static_cast<const __half*>(send_buf_ptr), send_counts, send_displs, static_cast<__half*>(recv_buf_ptr), recv_counts, recv_displs, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Alltoallv<Al::NCCLBackend>(static_cast<const unsigned char*>(send_buf_ptr), send_counts, send_displs, static_cast<unsigned char*>(recv_buf_ptr), recv_counts, recv_displs, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Alltoallv<Al::NCCLBackend>(static_cast<const long long int*>(send_buf_ptr), send_counts, send_displs, static_cast<long long int*>(recv_buf_ptr), recv_counts, recv_displs, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn send(&self, send_tensor: &dyn RawBaguaTensor, peer_rank: i32) {
        let communicator_ptr = self.comm_ptr;
        let tensor_ptr = send_tensor.data_ptr();
        let total_num_elem = send_tensor.num_elements_allocated();
        let nccl_tensor_type = send_tensor.dtype().to_nccl_datatype();

        unsafe {
            cpp::cpp!([tensor_ptr as "void *", total_num_elem as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t", peer_rank as "int"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Send<Al::NCCLBackend>(static_cast<float*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Send<Al::NCCLBackend>(static_cast<__half*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Send<Al::NCCLBackend>(static_cast<unsigned char*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Send<Al::NCCLBackend>(static_cast<long long int*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn recv(&self, recv_tensor: &mut dyn RawBaguaTensor, peer_rank: i32) {
        let communicator_ptr = self.comm_ptr;
        let tensor_ptr = recv_tensor.data_ptr();
        let total_num_elem = recv_tensor.num_elements_allocated();
        let nccl_tensor_type = recv_tensor.dtype().to_nccl_datatype();

        unsafe {
            cpp::cpp!([tensor_ptr as "void *", total_num_elem as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t", peer_rank as "int"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Recv<Al::NCCLBackend>(static_cast<float*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Recv<Al::NCCLBackend>(static_cast<__half*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Recv<Al::NCCLBackend>(static_cast<unsigned char*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Recv<Al::NCCLBackend>(static_cast<long long int*>(tensor_ptr), total_num_elem, peer_rank, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn allgather(
        &self,
        send_tensor: &dyn RawBaguaTensor,
        recv_tensor: &mut dyn RawBaguaTensor,
    ) {
        let communicator_ptr = self.comm_ptr;
        let send_tensor_ptr = send_tensor.data_ptr();
        assert_eq!(
            send_tensor.num_elements_allocated(),
            recv_tensor.num_elements_allocated()
        );
        assert_eq!(send_tensor.dtype(), recv_tensor.dtype());
        assert_eq!(
            send_tensor.num_elements_allocated() % self.nranks,
            0,
            "tensors must be aligned before using allgather"
        );
        let send_chunk_size = send_tensor.num_elements_allocated() / self.nranks;
        let nccl_tensor_type = send_tensor.dtype().to_nccl_datatype();

        let send_buf_ptr = send_tensor_ptr
            + self.rank as u64 * send_chunk_size as u64 * send_tensor.dtype().bytes() as u64;
        let recv_buf_ptr = recv_tensor.data_ptr();

        unsafe {
            cpp::cpp!([recv_buf_ptr as "void *", send_buf_ptr as "void *", send_chunk_size as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Allgather<Al::NCCLBackend>(static_cast<float*>(send_buf_ptr), static_cast<float*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Allgather<Al::NCCLBackend>(static_cast<__half*>(send_buf_ptr), static_cast<__half*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Allgather<Al::NCCLBackend>(static_cast<unsigned char*>(send_buf_ptr), static_cast<unsigned char*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Allgather<Al::NCCLBackend>(static_cast<long long int*>(send_buf_ptr), static_cast<long long int*>(recv_buf_ptr), send_chunk_size, *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }

    pub fn barrier(&self) {
        let communicator_ptr = self.comm_ptr;

        unsafe {
            cpp::cpp!([communicator_ptr as "Al::NCCLCommunicator *"]
            {
                Al::Barrier<Al::NCCLBackend>(*communicator_ptr);
            });
        }
    }

    pub fn allreduce(&self, tensor: &mut dyn RawBaguaTensor, op: BaguaReductionOp) {
        let communicator_ptr = self.comm_ptr;
        let tensor_ptr = tensor.data_ptr();
        let total_num_elem = tensor.num_elements_allocated();
        let nccl_tensor_type = tensor.dtype().to_nccl_datatype();

        unsafe {
            cpp::cpp!([tensor_ptr as "void *", total_num_elem as "size_t", communicator_ptr as "Al::NCCLCommunicator *", nccl_tensor_type as "ncclDataType_t", op as "uint8_t"]
            {
                if (nccl_tensor_type == ncclDataType_t::ncclFloat32) {
                    Al::Allreduce<Al::NCCLBackend>(static_cast<float*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclFloat16) {
                    Al::Allreduce<Al::NCCLBackend>(static_cast<__half*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclUint8) {
                    Al::Allreduce<Al::NCCLBackend>(static_cast<unsigned char*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), *communicator_ptr);
                } else if (nccl_tensor_type == ncclDataType_t::ncclInt64) {
                    Al::Allreduce<Al::NCCLBackend>(static_cast<long long int*>(tensor_ptr), total_num_elem, static_cast<Al::ReductionOperator>(op), *communicator_ptr);
                } else {
                    fputs("unsupport tensor data type.\n", stderr);
                    abort();
                }
            });
        }
    }
}
