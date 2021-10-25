extern crate env_logger;
extern crate libc;
extern crate log;

use std::ffi::CString;
use std::iter;

use bagua_core_internal::communicators::{BaguaCommOpConfig, BaguaSingleCommunicator};
use bagua_core_internal::datatypes::BaguaBucket;
use bagua_core_internal::datatypes::BaguaTensor;
use bagua_core_internal::telemetry::{
    BaguaCommCoreTelemetry, RegisterModelsRequest, TensorDeclaration,
};
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

#[no_mangle]
pub extern "C" fn bagua_tensor_c_create(
    ptr: u64,
    num_elem: usize,
    num_elem_allocated: usize,
    dtype_ptr: *const c_char,
    dtype_size: usize,
    device_id: usize,
) -> *mut BaguaTensorC {
    let obj = BaguaTensorC {
        inner: BaguaTensor::new(
            ptr,
            num_elem,
            num_elem_allocated,
            cstr_to_str(dtype_ptr, dtype_size),
            device_id,
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
    inplace: bool,
    align_bytes: usize,
) -> *mut BaguaBucketC {
    let tensor_ptr_slice: &[*mut BaguaTensorC] =
        unsafe { slice::from_raw_parts(tensors_ptr, tensors_len) };
    let mut tensors: Vec<&BaguaTensor> = Default::default();
    unsafe {
        for tensor_ptr in tensor_ptr_slice.iter() {
            tensors.push(&((*(*tensor_ptr)).inner));
        }
    };

    let new_bucket = BaguaBucket::new(
        tensors.as_slice(),
        cstr_to_str(name_ptr, name_size),
        inplace,
        align_bytes,
    );
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

pub struct BaguaCommOpConfigC {
    inner: BaguaCommOpConfig,
}

#[no_mangle]
pub extern "C" fn bagua_comm_op_config_c_create(
    communicator_internode: *mut BaguaSingleCommunicatorC,
    communicator_intranode: *mut BaguaSingleCommunicatorC,
    hierarchical: bool,
    average: bool,
    scattergather: bool,
    compression_ptr: *const c_char,
    compression_len: usize,
    is_decentralized: bool,
    peer_selection_mode_ptr: *const c_char,
    peer_selection_mode_len: usize,
    communication_interval: usize,
) -> *mut BaguaCommOpConfigC {
    let compression = if compression_ptr.is_null() {
        None
    } else {
        Some(cstr_to_str(compression_ptr, compression_len).to_string())
    };

    let obj = unsafe {
        BaguaCommOpConfigC {
            inner: BaguaCommOpConfig {
                communicator_internode: Some((*communicator_internode).inner.clone()),
                communicator_intranode: Some((*communicator_intranode).inner.clone()),
                hierarchical: hierarchical,
                average: average,
                scattergather: scattergather,
                compression: compression,
                is_decentralized: is_decentralized,
                peer_selection_mode: cstr_to_str(peer_selection_mode_ptr, peer_selection_mode_len)
                    .to_string(),
                communication_interval: communication_interval,
            },
        }
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_comm_op_config_c_destroy(ptr: &mut *mut BaguaCommOpConfigC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
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

pub struct TensorDeclarationC {
    inner: TensorDeclaration,
}

#[no_mangle]
pub extern "C" fn bagua_tensor_declaration_c_create(
    name_ptr: *const c_char,
    name_size: usize,
    num_elements: usize,
    dtype_ptr: *const c_char,
    dtype_size: usize,
) -> *mut TensorDeclarationC {
    let obj = TensorDeclarationC {
        inner: TensorDeclaration {
            name: cstr_to_str(name_ptr, name_size).to_string(),
            num_elements: num_elements,
            dtype: cstr_to_str(dtype_ptr, dtype_size).to_string(),
        },
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_tensor_declaration_c_destroy(ptr: &mut *mut TensorDeclarationC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn bagua_tensor_declaration_c_get_name(ptr: *mut TensorDeclarationC) -> *mut c_char {
    if ptr.is_null() {
        return std::ptr::null_mut();
    }

    let c_str = unsafe { CString::new((*ptr).inner.name.clone()).unwrap() };
    c_str.into_raw()
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

pub struct BaguaCommCoreTelemetryC {
    inner: BaguaCommCoreTelemetry,
}

#[no_mangle]
pub extern "C" fn bagua_comm_core_telemetry_c_create(
    server_addr_ptr: *const c_char,
    server_addr_len: usize,
) -> *mut BaguaCommCoreTelemetryC {
    let obj = BaguaCommCoreTelemetryC {
        inner: BaguaCommCoreTelemetry::new(cstr_to_str(server_addr_ptr, server_addr_len)),
    };

    Box::into_raw(Box::new(obj))
}

#[no_mangle]
pub extern "C" fn bagua_comm_core_telemetry_c_destroy(ptr: &mut *mut BaguaCommCoreTelemetryC) {
    if ptr.is_null() {
        return;
    }

    let _ = unsafe { Box::from_raw(*ptr) };

    *ptr = ::std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn bagua_comm_core_telemetry_c_register_models(
    ptr: *mut BaguaCommCoreTelemetryC,
    tensors_ptr: *const *mut TensorDeclarationC,
    tensors_len: usize,
    ordered_tensors: *mut usize,
    buckets_sizes: *mut usize,
) -> i32 {
    if ptr.is_null() {
        return -1;
    }

    let mut tensors: Vec<TensorDeclaration> = Default::default();
    unsafe {
        let slice: &[*mut TensorDeclarationC] = slice::from_raw_parts(tensors_ptr, tensors_len);
        for tensor_ptr in slice.iter() {
            tensors.push(((*(*tensor_ptr)).inner).clone());
        }
    }

    let req = RegisterModelsRequest {
        tensor_list: tensors.clone(),
    };

    let rsp = unsafe { (*ptr).inner.register_models(req) };
    let rsp = rsp.unwrap();

    let mut ordered_tensors_id = 0;
    let mut buckets_sizes_id = 0;
    for bucket in rsp.recommended_hyperparameters.buckets.iter() {
        for td in bucket.iter() {
            let td_id = tensors.iter().position(|x| x.name == td.name).unwrap();

            unsafe {
                let slice: &mut [usize] = slice::from_raw_parts_mut(ordered_tensors, tensors_len);
                slice[ordered_tensors_id] = td_id;

                ordered_tensors_id += 1;
            }
        }

        unsafe {
            let slice: &mut [usize] = slice::from_raw_parts_mut(buckets_sizes, tensors_len);
            slice[buckets_sizes_id] = bucket.len();

            buckets_sizes_id += 1;
        }
    }

    return 0;
}
