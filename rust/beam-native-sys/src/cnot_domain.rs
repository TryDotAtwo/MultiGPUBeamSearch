//! Thread-bound owner for the standalone CNOT CUDA domain C ABI.
//!
//! Host slices and optional CUDA views are borrowed only for a synchronous
//! ABI call; no device address is exposed as a Rust reference. This domain is
//! not yet connected to the public CNOT search or its multi-rank transport.
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
type DeviceOwners = unsafe extern "C" fn(
    *mut c_void,
    *const DeviceView,
    u64,
    *const DeviceView,
    *mut c_void,
) -> c_int;
type DeviceChildren = unsafe extern "C" fn(
    *mut c_void,
    *const DeviceView,
    u64,
    *const DeviceView,
    *const DeviceView,
    u32,
    u64,
    *const DeviceView,
    *mut c_void,
) -> c_int;

/// A non-owning CUDA address and its declared size in operation-specific elements.
/// Constructing this value does not validate or dereference the address.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DeviceView {
    data: *mut c_void,
    elements: u64,
    device: i32,
    reserved: u32,
}

impl DeviceView {
    pub fn from_raw(data: *mut c_void, elements: u64, device: i32) -> Self {
        Self {
            data,
            elements,
            device,
            reserved: 0,
        }
    }
}

/// Element type used for the physical CNOT move IDs of a borrowed expansion.
#[derive(Clone, Copy, Debug)]
#[repr(u32)]
pub enum DeviceMoveType {
    I64 = 0,
    U16 = 1,
}

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
    device_owners: Option<DeviceOwners>,
    create_device: Option<Create>,
    device_children: Option<DeviceChildren>,
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
            let device_owners =
                match library.get::<Version>(b"beam_cnot_domain_device_abi_version\0") {
                    Ok(version) => {
                        let actual = version();
                        if actual != 1 {
                            return Err(Error(format!(
                                "CNOT device ABI mismatch: required=1 available={actual}"
                            )));
                        }
                        Some(symbol!("beam_cnot_domain_owners_device", DeviceOwners))
                    }
                    Err(_) => None,
                };
            let create_device = library
                .get::<Create>(b"beam_cnot_domain_create_device\0")
                .ok()
                .map(|symbol| *symbol);
            let device_children = library
                .get::<DeviceChildren>(b"beam_cnot_domain_children_device\0")
                .ok()
                .map(|symbol| *symbol);
            if create_device.is_some() != device_children.is_some()
                || (create_device.is_some() && device_owners.is_none())
            {
                return Err(Error("incomplete CNOT device expansion ABI".into()));
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
                device_owners,
                create_device,
                device_children,
                _library: library,
            })))
        }
    }

    pub fn supports_device_owners(&self) -> bool {
        self.0.device_owners.is_some()
    }

    pub fn supports_device_expansion(&self) -> bool {
        self.0.create_device.is_some() && self.0.device_children.is_some()
    }

    pub fn create(&self, config: Config) -> Result<Domain> {
        self.create_with(config, self.0.create)
    }

    /// Create a context for borrowed device operations without host-ABI buffers.
    pub fn create_device(&self, config: Config) -> Result<Domain> {
        let create = self
            .0
            .create_device
            .ok_or_else(|| Error("CNOT domain library lacks device expansion support".into()))?;
        self.create_with(config, create)
    }

    fn create_with(&self, config: Config, create: Create) -> Result<Domain> {
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
        let code = unsafe { create(&native, &mut raw) };
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

/// Owns one CUDA device's stream and, for a full context, its internal buffers.
/// All operations and destruction remain on the creating thread.
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

    /// Compute owners directly in caller-owned CUDA storage on `stream`.
    ///
    /// # Safety
    /// Both views must describe live, correctly sized, contiguous int64 CUDA
    /// allocations on this domain's device, accessible from its CUDA context.
    /// Their used regions must not overlap. `stream` must be a valid stream in
    /// the same CUDA context (null is the default stream), with every input
    /// producer ordered before it.
    /// The caller retains the allocations and stream until this synchronous
    /// call returns and prevents concurrent conflicting access. Addresses are
    /// never dereferenced by Rust and must not be transferred to another rank.
    pub unsafe fn owners_device(
        &mut self,
        states: DeviceView,
        count: u64,
        owners: DeviceView,
        stream: *mut c_void,
    ) -> Result<()> {
        let call = self
            .api
            .device_owners
            .ok_or_else(|| Error("CNOT domain library lacks the device-owner extension".into()))?;
        // SAFETY: the caller supplies the documented CUDA lifetimes, extents
        // and stream ordering; this call synchronizes before returning.
        let code = unsafe { call(self.raw(), &states, count, &owners, stream) };
        self.api.check("device owners", code)
    }

    /// Expand CNOT children directly in caller-owned CUDA storage.
    ///
    /// # Safety
    /// All views must describe live, correctly sized, contiguous allocations
    /// accessible from this domain's CUDA context and remain valid until the
    /// call returns. Parents and children contain int64 matrix words; indices
    /// contain int64 parent IDs; moves contain elements of `move_type`.
    /// Every index and physical move must be in range, and the output region
    /// must not overlap any input. `stream` must be a live same-context stream
    /// (null is the default), with all input producers ordered before the call.
    /// The caller prevents concurrent conflicting access. CUDA addresses are
    /// not dereferenced by Rust and must not be transferred to another rank.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn expand_device(
        &mut self,
        parents: DeviceView,
        parent_count: u64,
        indices: DeviceView,
        moves: DeviceView,
        move_type: DeviceMoveType,
        child_count: u64,
        children: DeviceView,
        stream: *mut c_void,
    ) -> Result<()> {
        let call = self.api.device_children.ok_or_else(|| {
            Error("CNOT domain library lacks the device-expansion extension".into())
        })?;
        // SAFETY: the caller supplies the documented CUDA lifetimes, extents,
        // valid descriptors and stream ordering; native synchronizes on return.
        let code = unsafe {
            call(
                self.raw(),
                &parents,
                parent_count,
                &indices,
                &moves,
                move_type as u32,
                child_count,
                &children,
                stream,
            )
        };
        self.api.check("device expansion", code)
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
