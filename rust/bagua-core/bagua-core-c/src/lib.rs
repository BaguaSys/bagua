use bagua_core_internal::communicators::BaguaSingleCommunicator;
use std::ffi::CStr;
use std::os::raw::c_char;

pub struct BaguaSingleCommunicatorC {
    inner: BaguaSingleCommunicator,
}

pub extern "C" fn bagua_single_communicator_c_create(
    rank: usize,
    nranks: usize,
    device_id: usize,
    stream_ptr: u64,
    nccl_unique_id_str: *const c_char,
) -> *mut BaguaSingleCommunicatorC {
    let obj = BaguaSingleCommunicatorC {
        inner: bagua_core_internal::communicators::BaguaSingleCommunicator::new(
            rank,
            nranks,
            device_id,
            stream_ptr,
            unsafe { CStr::from_ptr(nccl_unique_id_str).to_str().unwrap() },
        ),
    };

    // into_raw turns the Box into a *mut, which the borrow checker
    // ignores, without calling its destructor.
    Box::into_raw(Box::new(obj))
}

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
pub extern "C" fn bagua_single_communicator_c_nranks(
    ptr: &mut *mut BaguaSingleCommunicatorC,
    nranks: *mut usize,
) -> i32 {
    // First, we **must** check to see if the pointer is null.
    if ptr.is_null() {
        // Do nothing.
        return -1;
    }

    unsafe {
        *nranks = (*(*ptr)).inner.nranks();
    }
    return 0;
}
