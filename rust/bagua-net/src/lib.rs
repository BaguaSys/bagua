#[macro_use]
extern crate lazy_static;

mod implement;
mod interface;
mod utils;

use ffi_convert::{AsRust, CDrop, CReprOf};
use implement::{nthread_per_socket_backend, tokio_backend};
use interface::{NCCLNetProperties, Net, SocketHandle};
use libc;
use std::sync::{Arc, Mutex};

pub struct BaguaNetC {
    inner: Arc<Mutex<Box<dyn Net>>>,
}

#[no_mangle]
pub extern "C" fn bagua_net_c_create() -> *mut BaguaNetC {
    let config = std::env::var("BAGUA_NET_IMPLEMENT")
        .unwrap_or("BASIC".to_owned())
        .to_uppercase();
    let bagua_net: Box<dyn Net> = match &config[..] {
        "TOKIO" => Box::new(tokio_backend::BaguaNet::new().unwrap()),
        "BASIC" => Box::new(nthread_per_socket_backend::BaguaNet::new().unwrap()),
        _ => {
            return std::ptr::null_mut();
        }
    };
    let obj = BaguaNetC {
        inner: Arc::new(Mutex::new(bagua_net)),
    };

    // into_raw turns the Box into a *mut, which the borrow checker
    // ignores, without calling its destructor.
    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_net_c_destroy(ptr: &mut *mut BaguaNetC) {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return;
    }

    // Now we know the pointer is non-null, we can continue. from_raw is the
    // inverse of into_raw: it turns the *mut Dramatic back into a
    // Box<Dramatic>. You must only call from_raw once per pointer.
    let obj: Box<BaguaNetC> = unsafe { Box::from_raw(*ptr) };

    // We don't *have* to do anything else; once obj goes out of scope, it will
    // be dropped.  I'm going to drop it explicitly, however, for clarity.
    drop(obj);

    // I am, however, going to null out the `ptr` we were passed just so the
    // calling code is less likely to accidentally re-use the pointer.
    *ptr = ::std::ptr::null_mut();
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_devices(ptr: *mut BaguaNetC, ndev: *mut i32) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        *ndev = (*ptr).inner.lock().unwrap().devices().unwrap() as i32;
    }
    return 0;
}

#[repr(C)]
#[derive(Debug, CReprOf, AsRust, CDrop)]
#[target_type(NCCLNetProperties)]
pub struct NCCLNetPropertiesC {
    pub name: *const libc::c_char,
    pub pci_path: *const libc::c_char,
    guid: u64,
    ptr_support: i32,
    speed: i32,
    port: i32,
    max_comms: i32,
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_get_properties(
    ptr: *mut BaguaNetC,
    dev_id: i32,
    props: *mut NCCLNetPropertiesC,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }
    if dev_id < 0 {
        return -2;
    }

    unsafe {
        let props_raw = (*ptr)
            .inner
            .lock()
            .unwrap()
            .get_properties(dev_id as usize)
            .unwrap();
        *props = NCCLNetPropertiesC::c_repr_of(props_raw).unwrap();
    }
    return 0;
}

#[repr(C)]
pub struct SocketHandleC {
    pub sockaddr: libc::sockaddr,
}

#[repr(C)]
pub struct SocketListenCommC {
    pub id: usize,
}

/// Error code
/// 0: success
/// -1: null pointer
/// -2: invalid parameter
/// -3: listen failed
#[no_mangle]
pub extern "C" fn bagua_net_c_listen(
    ptr: *mut BaguaNetC,
    dev_id: i32,
    socket_handle: *mut SocketHandleC,
    socket_listen_comm_id: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }
    if dev_id < 0 {
        return -2;
    }

    unsafe {
        let (handle, id) = match (*ptr).inner.lock().unwrap().listen(dev_id as usize) {
            Ok(result) => result,
            Err(_err) => return -3,
        };
        let (sockaddr, _) = handle.addr.as_ffi_pair();
        (*socket_handle).sockaddr = *sockaddr;
        *socket_listen_comm_id = id as usize;
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
/// -2: invalid parameter
/// -3: connect failed
#[no_mangle]
#[cfg(target_os = "linux")]
pub extern "C" fn bagua_net_c_connect(
    ptr: *mut BaguaNetC,
    dev_id: i32,
    socket_handle: *mut SocketHandleC,
    socket_send_comm_id: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }
    if dev_id < 0 {
        return -2;
    }

    unsafe {
        let sockaddr = (*socket_handle).sockaddr;

        let sockaddr = libc::sockaddr {
            sa_family: sockaddr.sa_family,
            sa_data: sockaddr.sa_data,
        };

        *socket_send_comm_id = match (*ptr).inner.lock().unwrap().connect(
            dev_id as usize,
            SocketHandle {
                addr: utils::from_libc_sockaddr(&sockaddr).unwrap(),
            },
        ) {
            Ok(id) => id,
            Err(_err) => return -3,
        }
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_accept(
    ptr: *mut BaguaNetC,
    listen_comm_id: usize,
    recv_comm_id: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        *recv_comm_id = (*ptr).inner.lock().unwrap().accept(listen_comm_id).unwrap();
    }
    return 0;
}

#[repr(C)]
pub struct Buffer {
    data: *mut u8,
    len: usize,
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_isend(
    ptr: *mut BaguaNetC,
    send_comm_id: usize,
    buf: Buffer,
    request_id: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        let data: &'static mut [u8] = std::slice::from_raw_parts_mut(buf.data, buf.len);
        *request_id = (*ptr)
            .inner
            .lock()
            .unwrap()
            .isend(send_comm_id, data)
            .unwrap();
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_irecv(
    ptr: *mut BaguaNetC,
    recv_comm_id: usize,
    buf: Buffer,
    request_id: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        let data: &'static mut [u8] = std::slice::from_raw_parts_mut(buf.data, buf.len);
        *request_id = (*ptr)
            .inner
            .lock()
            .unwrap()
            .irecv(recv_comm_id, data)
            .unwrap();
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
/// -2: invalid parameter
/// -3: bagua-net inner error
#[no_mangle]
pub extern "C" fn bagua_net_c_test(
    ptr: *mut BaguaNetC,
    request_id: usize,
    done: *mut bool,
    bytes: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }
    if done.is_null() {
        return -2;
    }

    unsafe {
        match (*ptr).inner.lock().unwrap().test(request_id) {
            Ok((let_done, let_bytes)) => {
                *done = let_done;
                if let_done && !bytes.is_null() {
                    *bytes = let_bytes;
                }
            }
            Err(err) => {
                tracing::warn!("{:?}", err);
                return -3;
            }
        }
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_close_send(ptr: *mut BaguaNetC, send_comm_id: usize) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        (*ptr)
            .inner
            .lock()
            .unwrap()
            .close_send(send_comm_id)
            .unwrap();
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_close_recv(ptr: *mut BaguaNetC, recv_comm_id: usize) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        (*ptr)
            .inner
            .lock()
            .unwrap()
            .close_recv(recv_comm_id)
            .unwrap();
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_net_c_close_listen(ptr: *mut BaguaNetC, listen_comm_id: usize) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        (*ptr)
            .inner
            .lock()
            .unwrap()
            .close_listen(listen_comm_id)
            .unwrap();
    }
    return 0;
}
