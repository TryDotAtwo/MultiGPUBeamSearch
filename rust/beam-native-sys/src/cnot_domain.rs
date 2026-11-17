//! Thread-bound owner for the standalone CNOT CUDA domain C ABI.
//!
//! Host slices are borrowed only for the duration of a synchronous ABI call;
//! no device address is exposed as a Rust reference. This domain is not yet
//! connected to the public CNOT search or its multi-rank transport.
use super::{Error, Result};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    mem::size_of,
    path::Path,
    ptr::NonNull,
    rc::Rc,
};

const ABI: u32 = 1;

#[derive(Clone, Copy, Debug)]
pub struct Config {
    pub device: i32,
    pub n: i32,
    pub world: i32,
    pub parent_capacity: u64,
    pub child_capacity: u64,
    pub closed_capacity: u64,
}

#[repr(C)]
struct NativeConfig {
    abi_version: u32,
    struct_size: u32,
    device: i32,
    n: i32,
    world: i32,
    reserved: u32,
    parent_capacity: u64,
    child_capacity: u64,
    closed_capacity: u64,
}

type Version = unsafe extern "C" fn() -> u32;
type LastError = unsafe extern "C" fn() -> *const c_char;
type Create = unsafe extern "C" fn(*const NativeConfig, *mut *mut c_void) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void) -> c_int;
type Upload = unsafe extern "C" fn(*mut c_void, *const i64, u64, u64) -> c_int;
type Descriptors = unsafe extern "C" fn(*mut c_void, *const i64, u64, *const i64, u64) -> c_int;
type Stage = unsafe extern "C" fn(*mut c_void) -> c_int;
type Select = unsafe extern "C" fn(*mut c_void, *mut u64) -> c_int;
type Download = unsafe extern "C" fn(*mut c_void, *mut i64, u64) -> c_int;

struct Bindings {
    _library: Library,
    last_error: LastError,
    create: Create,
    destroy: Destroy,
    parents: Upload,
    closed: Upload,
    descriptors: Descriptors,
    expand: Stage,
    owners: Stage,
    select: Select,
    download_children: Download,
    download_owners: Download,
    download_selected: Download,
}

impl Bindings {
    fn check(&self, operation: &str, result: c_int) -> Result<()> {
        if result == 0 {
            return Ok(());
        }
        // SAFETY: the loaded ABI guarantees a thread-local NUL-terminated
        // message, valid until the next ABI call on this same thread.
        let message = unsafe {
            let ptr = (self.last_error)();
            if ptr.is_null() {
                "unknown native error".into()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Err(Error(format!(
            "CNOT domain {operation} failed ({result}): {message}"
        )))
    }
}

/// The dynamically loaded library remains alive while any domain exists.
#[derive(Clone)]
pub struct DomainLibrary(Rc<Bindings>);

impl DomainLibrary {
    /// Load the standalone native domain without loading Python or Torch.
    ///
    /// # Safety
    /// `path` must identify a trusted library implementing the versioned
    /// `cnot_owned_domain.h` ABI, including its pointer and lifetime rules.
    pub unsafe fn load(path: &Path) -> Result<Self> {
        // SAFETY: the caller guarantees the library implements this ABI; the
        // Rc keeps it loaded until after the last domain is destroyed.
        unsafe {
            let library =
                Library::new(path).map_err(|e| Error(format!("load {}: {e}", path.display())))?;
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|e| Error(e.to_string()))?
                };
            }
            let version = symbol!("beam_cnot_domain_abi_version", Version)();
            if version != ABI {
                return Err(Error(format!(
                    "CNOT domain ABI mismatch: required={ABI} available={version}"
                )));
            }
            Ok(Self(Rc::new(Bindings {
                last_error: symbol!("beam_cnot_domain_error", LastError),
                create: symbol!("beam_cnot_domain_create", Create),
                destroy: symbol!("beam_cnot_domain_destroy", Destroy),
                parents: symbol!("beam_cnot_domain_set_parents", Upload),
                closed: symbol!("beam_cnot_domain_set_closed", Upload),
                descriptors: symbol!("beam_cnot_domain_set_descriptors", Descriptors),
                expand: symbol!("beam_cnot_domain_expand", Stage),
                owners: symbol!("beam_cnot_domain_owners", Stage),
                select: symbol!("beam_cnot_domain_first_unseen", Select),
                download_children: symbol!("beam_cnot_domain_download_children", Download),
                download_owners: symbol!("beam_cnot_domain_download_owners", Download),
                download_selected: symbol!("beam_cnot_domain_download_selected", Download),
                _library: library,
            })))
        }
    }

    pub fn create(&self, config: Config) -> Result<Domain> {
        if !(2..=256).contains(&config.n) {
            return Err(Error("invalid CNOT qubit count".into()));
        }
        let stride = config.n as usize * ((config.n as usize + 63) / 64);
        let native = NativeConfig {
            abi_version: ABI,
            struct_size: size_of::<NativeConfig>() as u32,
            device: config.device,
            n: config.n,
            world: config.world,
            reserved: 0,
            parent_capacity: config.parent_capacity,
            child_capacity: config.child_capacity,
            closed_capacity: config.closed_capacity,
        };
        let mut raw = std::ptr::null_mut();
        // SAFETY: config and output are valid for this blocking call; the
        // checked ABI either returns an owned context or a failure.
        let code = unsafe { (self.0.create)(&native, &mut raw) };
        self.0.check("create", code)?;
        let ptr =
            NonNull::new(raw).ok_or_else(|| Error("CNOT domain create returned null".into()))?;
        Ok(Domain {
            api: self.0.clone(),
            ptr: Some(ptr),
            stride,
            child_count: 0,
        })
    }
}

/// Owns one CUDA device's buffers and stream on its creating thread.
///
/// ```compile_fail
/// fn requires_send<T: Send>() {}
/// requires_send::<beam_native_sys::cnot_domain::Domain>();
/// ```
pub struct Domain {
    api: Rc<Bindings>,
    ptr: Option<NonNull<c_void>>,
    stride: usize,
    child_count: usize,
}

impl Domain {
    fn raw(&self) -> *mut c_void {
        self.ptr.expect("domain only closes when consumed").as_ptr()
    }

    fn upload(&mut self, data: &[i64], operation: &str, call: Upload) -> Result<()> {
        if data.len() % self.stride != 0 {
            return Err(Error(format!(
                "CNOT {operation} words do not form whole states"
            )));
        }
        let count = data.len() / self.stride;
        // SAFETY: the host slice is readable for its length and remains alive
        // until the ABI finishes its synchronous copy.
        let code = unsafe { call(self.raw(), data.as_ptr(), data.len() as u64, count as u64) };
        self.api.check(operation, code)
    }

    pub fn set_parents(&mut self, states: &[i64]) -> Result<()> {
        self.upload(states, "parents", self.api.parents)
    }

    pub fn set_closed(&mut self, states: &[i64]) -> Result<()> {
        self.upload(states, "closed", self.api.closed)
    }

    pub fn set_descriptors(&mut self, indices: &[i64], moves: &[i64]) -> Result<()> {
        // SAFETY: both immutable slices remain live through the synchronous
        // upload. Native validates lengths, parents, moves and capacity.
        let code = unsafe {
            (self.api.descriptors)(
                self.raw(),
                indices.as_ptr(),
                indices.len() as u64,
                moves.as_ptr(),
                moves.len() as u64,
            )
        };
        self.api.check("descriptors", code)?;
        self.child_count = indices.len();
        Ok(())
    }

    pub fn expand(&mut self) -> Result<()> {
        // SAFETY: the context is valid and this borrow excludes other calls.
        let code = unsafe { (self.api.expand)(self.raw()) };
        self.api.check("expand", code)
    }

    pub fn owners(&mut self) -> Result<Vec<i64>> {
        // SAFETY: the context is valid and this borrow excludes other calls.
        let code = unsafe { (self.api.owners)(self.raw()) };
        self.api.check("owners", code)?;
        self.download(self.api.download_owners, self.child_count, "owners")
    }

    pub fn children(&mut self) -> Result<Vec<i64>> {
        let words = self
            .child_count
            .checked_mul(self.stride)
            .ok_or_else(|| Error("CNOT children word count overflow".into()))?;
        self.download(self.api.download_children, words, "children")
    }

    pub fn first_unseen(&mut self) -> Result<Vec<i64>> {
        let mut count = 0u64;
        // SAFETY: count is a writable host value and remains alive during the
        // native call, which synchronizes its data-dependent count.
        let code = unsafe { (self.api.select)(self.raw(), &mut count) };
        self.api.check("first unseen", code)?;
        let count =
            usize::try_from(count).map_err(|_| Error("CNOT selected count overflow".into()))?;
        if count > self.child_count {
            return Err(Error("CNOT selected count exceeds child count".into()));
        }
        self.download(self.api.download_selected, count, "selected")
    }

    fn download(&mut self, call: Download, count: usize, operation: &str) -> Result<Vec<i64>> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| Error(format!("CNOT {operation} allocation: {e}")))?;
        values.resize(count, 0);
        // SAFETY: values is initialized and writable for count elements until
        // this synchronous host download returns.
        let code = unsafe { call(self.raw(), values.as_mut_ptr(), values.len() as u64) };
        self.api.check(operation, code)?;
        Ok(values)
    }

    /// Explicitly release the CUDA resources, surfacing a native cleanup error.
    pub fn close(mut self) -> Result<()> {
        self.release()
    }

    fn release(&mut self) -> Result<()> {
        let Some(ptr) = self.ptr.take() else {
            return Ok(());
        };
        // SAFETY: this unique context is destroyed once on its owning thread;
        // Rc retains the library for the full call.
        let code = unsafe { (self.api.destroy)(ptr.as_ptr()) };
        self.api.check("destroy", code)
    }
}

impl Drop for Domain {
    fn drop(&mut self) {
        let _ = self.release();
    }
}
