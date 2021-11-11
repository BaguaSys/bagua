extern crate env_logger;
extern crate libc;
extern crate log;

use std::ffi::CString;
use std::iter;

use bagua_core_internal::communicators::BaguaSingleCommunicator;
use bagua_core_internal::datatypes::{BaguaBucket, BaguaTensor, BaguaTensorDtype};
use bagua_core_internal::BaguaCommBackend;
use libc::c_char;
use std::{slice, str};

pub struct BaguaSingleCommunicatorC {
    inner: BaguaSingleCommunicator,
}

pub fn cstr_to_str(c_s: *const c_char, size: usize) -> &'static str {
    unsafe { str::from_utf8_unchecked(slice::from_raw_parts(c_s as *const u8, size)) }
}

#[no_mangle]
pub extern "C" fn bagua_single_communicator_c_create(
    rank: usize,
    nranks: usize,
    device_id: usize,
    stream_ptr: u64,
    nccl_unique_id_ptr: *const c_char,
    nccl_unique_id_size: usize,
) -> *mut BaguaSingleCommunicatorC {
    let obj = BaguaSingleCommunicatorC {
        inner: bagua_core_internal::communicators::BaguaSingleCommunicator::new(
            rank,
            nranks,
            device_id,
            stream_ptr,
            cstr_to_str(nccl_unique_id_ptr, nccl_unique_id_size),
        ),
    };

    // into_raw turns the Box into a *mut, which the borrow checker
    // ignores, without calling its destructor.
    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_single_communicator_c_destroy(ptr: &mut *mut BaguaSingleCommunicatorC) {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return;
    }

    // Now we know the pointer is non-null, we can continue. from_raw is the
    // inverse of into_raw: it turns the *mut Dramatic back into a
    // Box<Dramatic>. You must only call from_raw once per pointer.
    let obj: Box<BaguaSingleCommunicatorC> = unsafe { Box::from_raw(*ptr) };

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
pub extern "C" fn bagua_single_communicator_c_nranks(
    ptr: *mut BaguaSingleCommunicatorC,
    nranks: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        *nranks = (*ptr).inner.nranks();
    }
    return 0;
}

/// Error code
/// 0: success
/// -1: null pointer
#[no_mangle]
pub extern "C" fn bagua_single_communicator_c_rank(
    ptr: *mut BaguaSingleCommunicatorC,
    rank: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        *rank = (*ptr).inner.rank();
    }
    return 0;
}

pub struct BaguaTensorC {
    inner: BaguaTensor,
}

#[repr(u32)]
pub enum BaguaTensorDtypeFFI {
    F32,
    F16,
    U8,
    I64,
    U64,
}

impl BaguaTensorDtypeFFI {
    pub fn inner(&self) -> BaguaTensorDtype {
        match self {
            BaguaTensorDtypeFFI::F32 => BaguaTensorDtype::F32,
            BaguaTensorDtypeFFI::F16 => BaguaTensorDtype::F16,
            BaguaTensorDtypeFFI::U8 => BaguaTensorDtype::U8,
            BaguaTensorDtypeFFI::I64 => BaguaTensorDtype::I64,
            BaguaTensorDtypeFFI::U64 => BaguaTensorDtype::U64,
        }
    }
}

#[no_mangle]
pub extern "C" fn bagua_tensor_c_create(
    name_ptr: *const c_char,
    name_size: usize,
    device_id: usize,
    data_ptr: u64,
    num_elem: usize,
    dtype: BaguaTensorDtypeFFI,
    ready_cuda_event_ptr: u64,
) -> *mut BaguaTensorC {
    let obj = BaguaTensorC {
        inner: BaguaTensor::new(
            cstr_to_str(name_ptr, name_size).to_string(),
            device_id,
            data_ptr,
            num_elem,
            dtype.inner(),
            ready_cuda_event_ptr,
        ),
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_tensor_c_destroy(ptr: &mut *mut BaguaTensorC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

pub struct BaguaBucketC {
    inner: BaguaBucket,
}

#[no_mangle]
pub extern "C" fn bagua_bucket_c_create(
    tensors_ptr: *const *mut BaguaTensorC,
    tensors_len: usize,
    name_ptr: *const c_char,
    name_size: usize,
) -> *mut BaguaBucketC {
    let tensor_ptr_slice: &[*mut BaguaTensorC] =
        unsafe { slice::from_raw_parts(tensors_ptr, tensors_len) };
    let mut tensors: Vec<&BaguaTensor> = Default::default();
    unsafe {
        for tensor_ptr in tensor_ptr_slice.iter() {
            tensors.push(&((*(*tensor_ptr)).inner));
        }
    };

    let new_bucket = BaguaBucket::new(tensors.as_slice(), cstr_to_str(name_ptr, name_size));
    let new_bucket = match new_bucket {
        Ok(bucket) => bucket,
        Err(error) => {
            println!("BaguaBucket::new failed, error={:?}", error);
            return std::ptr::null_mut();
        }
    };

    let obj = BaguaBucketC { inner: new_bucket };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_bucket_c_destroy(ptr: &mut *mut BaguaBucketC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn bagua_bucket_c_append_centralized_synchronous_op(
    ptr: *mut BaguaBucketC,
    communicator_internode: *mut BaguaSingleCommunicatorC,
    communicator_intranode: *mut BaguaSingleCommunicatorC,
    hierarchical: bool,
    average: bool,
    scattergather: bool,
) {
    if ptr.is_null() {
        return;
    }

    unsafe {
        (*ptr).inner.append_centralized_synchronous_op(
            Some(&((*communicator_internode).inner)),
            Some(&((*communicator_intranode).inner)),
            hierarchical,
            average,
            scattergather,
            None,
        );
    }
}

pub struct BaguaCommBackendC {
    inner: BaguaCommBackend,
}

#[no_mangle]
pub extern "C" fn bagua_comm_backend_c_create(
    schedule_channel_cap: usize,
    device_id: usize,
) -> *mut BaguaCommBackendC {
    let obj = BaguaCommBackendC {
        inner: BaguaCommBackend::new(schedule_channel_cap, device_id),
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_comm_backend_c_destroy(ptr: &mut *mut BaguaCommBackendC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn bagua_comm_backend_c_register_ordered_buckets(
    ptr: *mut BaguaCommBackendC,
    buckets_ptr: *const *mut BaguaBucketC,
    buckets_len: usize,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let mut buckets: Vec<&BaguaBucket> = Default::default();
    unsafe {
        let slice: &[*mut BaguaBucketC] = slice::from_raw_parts(buckets_ptr, buckets_len);
        for bucket_ptr in slice.iter() {
            buckets.push(&((*(*bucket_ptr)).inner));
        }
    }

    let ret = unsafe { (*ptr).inner.register_ordered_buckets(buckets.as_slice()) };
    match ret {
        Ok(_) => {}
        Err(err) => {
            println!("register_ordered_buckets failed, err={:?}", err);
            return -1;
        }
    };

    return 0;
}

#[no_mangle]
pub extern "C" fn bagua_comm_backend_c_mark_communication_ready(
    ptr: *mut BaguaCommBackendC,
    bagua_tensor: *mut BaguaTensorC,
    ready_cuda_event_ptr: u64,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let ret = unsafe {
        (*ptr)
            .inner
            .mark_communication_ready(&((*bagua_tensor).inner), ready_cuda_event_ptr)
    };
    match ret {
        Ok(_) => {}
        Err(err) => {
            println!("mark_communication_ready failed, err={:?}", err);
            return -1;
        }
    };

    return 0;
}

#[no_mangle]
pub extern "C" fn bagua_comm_backend_c_wait_pending_comm_ops(ptr: *mut BaguaCommBackendC) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let ret = unsafe { (*ptr).inner.wait_pending_comm_ops() };
    match ret {
        Ok(_) => {}
        Err(err) => {
            println!("mark_communication_ready failed, err={:?}", err);
            return -1;
        }
    };

    return 0;
}

#[no_mangle]
pub extern "C" fn cstring_free(s: *mut c_char) {
    unsafe {
        if s.is_null() {
            return;
        }
        CString::from_raw(s);
    }
}
