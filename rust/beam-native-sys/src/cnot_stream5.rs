//! Owned, thread-bound handle for the existing versioned CNOT Stream 5 ABI.
//!
//! This module does not load CUDA until explicitly requested, and does not
//! accept device pointers. Its native owner retains the initialization device
//! and restores the caller's current device after cleanup. The owner remains
//! bound to its creating thread. The native library stays loaded until every
//! communicator created from it has been closed or dropped.
use super::{Error, Result};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

const ABI: c_int = 1;
const OWNED_ABI: c_int = 1;
const MAX_ID_BYTES: usize = 4096;
const MAX_WORLD: u32 = 65536;

type Version = unsafe extern "C" fn() -> c_int;
type IdSize = unsafe extern "C" fn() -> c_int;
type CreateId = unsafe extern "C" fn(*mut c_void) -> c_int;
type Init = unsafe extern "C" fn(*mut *mut c_void, *const c_void, c_int, c_int) -> c_int;
type Close = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type LastError = unsafe extern "C" fn() -> *const c_char;

struct Bindings {
    _library: Library,
    id_size: usize,
    create_id: CreateId,
    init: Init,
    close: Close,
    last_error: LastError,
}

impl Bindings {
    fn error(&self, operation: &str, code: c_int) -> Error {
        // SAFETY: the versioned native ABI returns a thread-local, NUL-terminated
        // error string, valid until the next call on this same thread.
        let ptr = unsafe { (self.last_error)() };
        let detail = if ptr.is_null() {
            "unknown native error".to_string()
        } else {
            unsafe { CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned()
        };
        Error(format!(
            "CNOT Stream 5 {operation} failed ({code}): {detail}"
        ))
    }
}

/// A dynamically loaded ABI. Rc makes the library and its contexts thread-bound.
#[derive(Clone)]
pub struct Stream5Library(Rc<Bindings>);

impl Stream5Library {
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: the caller selects a native library implementing this ABI.
        // The owned Library remains alive in every Rc<Bindings> handle.
        unsafe {
            let library = Library::new(path)
                .map_err(|error| Error(format!("load {}: {error}", path.display())))?;
            let version = *library
                .get::<Version>(b"cnot_s5_abi\0")
                .map_err(|error| Error(error.to_string()))?;
            let actual = version();
            if actual != ABI {
                return Err(Error(format!(
                    "CNOT Stream 5 ABI mismatch: required={ABI} available={actual}"
                )));
            }
            let owned_version = *library
                .get::<Version>(b"cnot_s5_owned_abi\0")
                .map_err(|error| Error(format!("CNOT Stream 5 owned ABI unavailable: {error}")))?;
            let owned_actual = owned_version();
            if owned_actual != OWNED_ABI {
                return Err(Error(format!(
                    "CNOT Stream 5 owned ABI mismatch: required={OWNED_ABI} available={owned_actual}"
                )));
            }
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|error| Error(error.to_string()))?
                };
            }
            let size = symbol!("cnot_s5_id_bytes", IdSize)();
            if size <= 0 || size as usize > MAX_ID_BYTES {
                return Err(Error(format!(
                    "invalid CNOT Stream 5 identity size: {size}"
                )));
            }
            Ok(Self(Rc::new(Bindings {
                id_size: size as usize,
                create_id: symbol!("cnot_s5_id", CreateId),
                init: symbol!("cnot_s5_owned_init", Init),
                close: symbol!("cnot_s5_owned_close", Close),
                last_error: symbol!("cnot_s5_error", LastError),
                _library: library,
            })))
        }
    }

    pub fn unique_id(&self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.0.id_size];
        // SAFETY: the checked ABI writes exactly id_size bytes into this buffer.
        let code = unsafe { (self.0.create_id)(bytes.as_mut_ptr().cast()) };
        if code != 0 {
            return Err(self.0.error("create identity", code));
        }
        Ok(bytes)
    }

    pub fn init(&self, id: &[u8], rank: u32, world: u32) -> Result<Stream5Communicator> {
        if id.len() != self.0.id_size {
            return Err(Error(format!(
                "CNOT Stream 5 identity size: required={} available={}",
                self.0.id_size,
                id.len()
            )));
        }
        if world == 0 || world > MAX_WORLD || rank >= world {
            return Err(Error(format!(
                "invalid CNOT Stream 5 rank {rank} of {world}"
            )));
        }
        let mut raw = std::ptr::null_mut();
        // SAFETY: the ID has the native ABI's exact size; raw is a valid out
        // pointer. Both remain alive for this synchronous native call.
        let code =
            unsafe { (self.0.init)(&mut raw, id.as_ptr().cast(), rank as c_int, world as c_int) };
        if code != 0 {
            return Err(self.0.error("initialize communicator", code));
        }
        let ptr = NonNull::new(raw)
            .ok_or_else(|| Error("CNOT Stream 5 returned a null communicator".into()))?;
        Ok(Stream5Communicator {
            bindings: self.0.clone(),
            ptr: Some(ptr),
        })
    }
}

/// Owns one NCCL communicator; never send it to another thread or process.
pub struct Stream5Communicator {
    bindings: Rc<Bindings>,
    ptr: Option<NonNull<c_void>>,
}

impl Stream5Communicator {
    pub fn is_closed(&self) -> bool {
        self.ptr.is_none()
    }

    pub fn close(&mut self, abort: bool) -> Result<()> {
        let Some(ptr) = self.ptr.take() else {
            return Ok(());
        };
        // SAFETY: this communicator was returned by init on this thread and is
        // consumed exactly once. The library outlives the call via bindings.
        let code = unsafe { (self.bindings.close)(ptr.as_ptr(), c_int::from(abort)) };
        if code != 0 {
            return Err(self.bindings.error("close communicator", code));
        }
        Ok(())
    }
}

impl Drop for Stream5Communicator {
    fn drop(&mut self) {
        // Explicit close remains preferred; abandonment aborts the owned rank.
        let _ = self.close(true);
    }
}
