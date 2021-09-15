use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum BaguaNetError {
    #[error("io error")]
    IOError(String),
    #[error("tcp error")]
    TCPError(String),
    #[error("inner error")]
    InnerError(String),
}

#[derive(Debug)]
pub struct NCCLNetProperties {
    pub name: String,
    pub pci_path: String,
    pub guid: u64,
    pub ptr_support: i32, // NCCL_PTR_HOST or NCCL_PTR_HOST|NCCL_PTR_CUDA
    pub speed: i32,       // Port speed in Mbps.
    pub port: i32,
    pub max_comms: i32,
}

#[derive(Debug)]
pub struct SocketHandle {
    pub addr: nix::sys::socket::SockAddr,
}

pub type SocketListenCommID = usize;
pub type SocketSendCommID = usize;
pub type SocketRecvCommID = usize;
pub type SocketRequestID = usize;

pub trait Net {
    fn devices(&self) -> Result<usize, BaguaNetError>;

    fn get_properties(&self, dev_id: usize) -> Result<NCCLNetProperties, BaguaNetError>;

    fn listen(
        &mut self,
        dev_id: usize,
    ) -> Result<(SocketHandle, SocketListenCommID), BaguaNetError>;

    fn connect(
        &mut self,
        _dev_id: usize,
        socket_handle: SocketHandle,
    ) -> Result<SocketSendCommID, BaguaNetError>;

    fn accept(
        &mut self,
        listen_comm_id: SocketListenCommID,
    ) -> Result<SocketRecvCommID, BaguaNetError>;

    fn isend(
        &mut self,
        send_comm_id: SocketSendCommID,
        data: &'static [u8],
    ) -> Result<SocketRequestID, BaguaNetError>;

    fn irecv(
        &mut self,
        recv_comm_id: SocketRecvCommID,
        data: &'static mut [u8],
    ) -> Result<SocketRequestID, BaguaNetError>;

    fn test(&mut self, request_id: SocketRequestID) -> Result<(bool, usize), BaguaNetError>;

    fn close_send(&mut self, send_comm_id: SocketSendCommID) -> Result<(), BaguaNetError>;

    fn close_recv(&mut self, recv_comm_id: SocketRecvCommID) -> Result<(), BaguaNetError>;

    fn close_listen(&mut self, listen_comm_id: SocketListenCommID) -> Result<(), BaguaNetError>;
}
