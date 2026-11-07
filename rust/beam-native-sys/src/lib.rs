//! The only crate permitted to dereference native FFI pointers.
use libloading::Library;
use std::{
    ffi::{c_char, c_void, CString},
    marker::PhantomData,
    path::Path,
    ptr::NonNull,
    rc::Rc,
    sync::Arc,
    time::Duration,
};

pub const ABI_VERSION: u32 = 1;
const OK: i32 = 0;
const TIMEOUT: i32 = 5;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);
type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
struct Job {
    abi_version: u32,
    struct_size: u32,
    sequence: u64,
    state: *const u8,
    state_len: u64,
    directory: *const c_char,
    directory_len: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Status {
    abi_version: u32,
    struct_size: u32,
    pub phase: u32,
    pub result: u32,
    pub completed_jobs: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Layout {
    abi_version: u32,
    struct_size: u32,
    pub state_len: u32,
    pub storage_len: u32,
    pub state_alignment: u32,
    pub move_count: u32,
    pub candidate_bytes: u32,
    pub candidate_alignment: u32,
    pub hash_bytes: u32,
    pub hash_alignment: u32,
}
type GetLayout = unsafe extern "C" fn(*mut Layout) -> i32;
type Create = unsafe extern "C" fn(u32, *mut *mut c_void) -> i32;
type Prepare = unsafe extern "C" fn(*mut c_void, i32, *const *const c_char) -> i32;
type Submit = unsafe extern "C" fn(*mut c_void, *const Job) -> i32;
type Wait = unsafe extern "C" fn(*mut c_void, u64) -> i32;
type GetStatus = unsafe extern "C" fn(*mut c_void, *mut Status) -> i32;
type Cancel = unsafe extern "C" fn(*mut c_void) -> i32;
type GetError = unsafe extern "C" fn(*mut c_void, *mut c_char, u64, *mut u64) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void) -> i32;
type Run = unsafe extern "C" fn(i32, *const *const c_char) -> i32;

struct Bindings {
    _library: Library,
    create: Create,
    prepare: Prepare,
    submit: Submit,
    wait: Wait,
    status: GetStatus,
    cancel: Cancel,
    error: GetError,
    destroy: Destroy,
    run: Run,
    layout: GetLayout,
}
#[derive(Clone)]
pub struct NativeLibrary(Arc<Bindings>);
impl NativeLibrary {
    /// Load only in a dedicated rank process. Importing beam-core/runtime never
    /// loads this library, CUDA, NCCL, or model weights.
    pub fn load(path: &Path) -> Result<Self> {
        // SAFETY: expected, versioned local native library. Symbols stay alive
        // through the Arc held by every context; no context outlives dlclose.
        unsafe {
            let library =
                Library::new(path).map_err(|e| Error(format!("load {}: {e}", path.display())))?;
            let version = library
                .get::<unsafe extern "C" fn() -> u32>(b"beam_native_abi_version\0")
                .map_err(|e| Error(e.to_string()))?();
            if version != ABI_VERSION {
                return Err(Error(format!(
                    "native ABI mismatch: required={ABI_VERSION} available={version}"
                )));
            }
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(|e| Error(e.to_string()))?
                };
            }
            let bindings = Bindings {
                create: symbol!("beam_context_create", Create),
                prepare: symbol!("beam_context_prepare", Prepare),
                submit: symbol!("beam_context_submit", Submit),
                wait: symbol!("beam_context_wait", Wait),
                status: symbol!("beam_context_status", GetStatus),
                cancel: symbol!("beam_context_cancel", Cancel),
                error: symbol!("beam_context_error", GetError),
                destroy: symbol!("beam_context_destroy", Destroy),
                run: symbol!("beam_native_main", Run),
                layout: symbol!("beam_native_layout", GetLayout),
                _library: library,
            };
            Ok(Self(Arc::new(bindings)))
        }
    }
    pub fn layout(&self) -> Result<Layout> {
        let mut layout = Layout {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Layout>() as u32,
            ..Layout::default()
        };
        let result = unsafe { (self.0.layout)(&mut layout) };
        if result != OK {
            return Err(Error(format!("native layout query failed: {result}")));
        }
        Ok(layout)
    }
    pub fn context(&self) -> Result<Context> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: out pointer is writable and ABI has been checked at load.
        let code = unsafe { (self.0.create)(ABI_VERSION, &mut ptr) };
        if code != OK {
            return Err(Error(format!("native context creation failed: {code}")));
        }
        let ptr = NonNull::new(ptr).ok_or_else(|| Error("native create returned null".into()))?;
        Ok(Context {
            ptr,
            api: self.0.clone(),
            _thread_owner: PhantomData,
        })
    }
    pub fn run(&self, args: &[String]) -> Result<i32> {
        let storage = arguments(args)?;
        let views = argument_views(&storage);
        // SAFETY: strings are owned until the blocking coarse call returns.
        Ok(unsafe { (self.0.run)(storage.len() as i32, views.as_ptr()) })
    }
}
fn arguments(args: &[String]) -> Result<Vec<CString>> {
    if args.is_empty() || args.len() > 4096 {
        return Err(Error("invalid native argument count".into()));
    }
    args.iter()
        .map(|arg| CString::new(arg.as_bytes()).map_err(|_| Error("argument contains NUL".into())))
        .collect()
}
fn argument_views(storage: &[CString]) -> Vec<*const c_char> {
    storage
        .iter()
        .map(|arg| arg.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect()
}
pub struct Context {
    ptr: NonNull<c_void>,
    api: Arc<Bindings>,
    // A context cannot be sent/shared across Rust threads or processes.
    _thread_owner: PhantomData<Rc<()>>,
}
impl Context {
    fn check(&self, code: i32) -> Result<()> {
        if code == OK {
            return Ok(());
        }
        let mut required = 0;
        // SAFETY: context remains owned, first call is the documented size query.
        unsafe {
            (self.api.error)(self.ptr.as_ptr(), std::ptr::null_mut(), 0, &mut required);
        }
        let mut message = vec![0u8; required.clamp(1, 1 << 20) as usize];
        // SAFETY: destination covers the supplied capacity; no native pointer is
        // turned into a Rust reference and nothing escapes this operation.
        let copied = unsafe {
            (self.api.error)(
                self.ptr.as_ptr(),
                message.as_mut_ptr().cast(),
                message.len() as u64,
                &mut required,
            )
        };
        if copied == OK && message.first() != Some(&0) {
            message.truncate(
                message
                    .iter()
                    .position(|&c| c == 0)
                    .unwrap_or(message.len()),
            );
            Err(Error(format!(
                "native error {code}: {}",
                String::from_utf8_lossy(&message)
            )))
        } else {
            Err(Error(format!("native error {code}")))
        }
    }
    pub fn prepare(&mut self, args: &[String]) -> Result<()> {
        let storage = arguments(args)?;
        let views = argument_views(&storage);
        // SAFETY: native prepare copies each argument before returning.
        self.check(unsafe {
            (self.api.prepare)(self.ptr.as_ptr(), storage.len() as i32, views.as_ptr())
        })
    }
    pub fn submit(&mut self, sequence: u64, state: &[u8], directory: &Path) -> Result<()> {
        let directory = directory
            .to_str()
            .ok_or_else(|| Error("output path is not UTF-8".into()))?;
        let job = Job {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Job>() as u32,
            sequence,
            state: state.as_ptr(),
            state_len: state.len() as u64,
            directory: directory.as_ptr().cast(),
            directory_len: directory.len() as u64,
        };
        // SAFETY: both borrowed slices outlive this call; native submit copies
        // the small job descriptor and never retains a Python/Rust address.
        self.check(unsafe { (self.api.submit)(self.ptr.as_ptr(), &job) })
    }
    pub fn wait(&self, timeout: Duration) -> Result<bool> {
        let milliseconds =
            u64::try_from(timeout.as_millis()).map_err(|_| Error("timeout overflow".into()))?;
        let code = unsafe { (self.api.wait)(self.ptr.as_ptr(), milliseconds) };
        if code == TIMEOUT {
            Ok(false)
        } else {
            self.check(code)?;
            Ok(true)
        }
    }
    pub fn status(&self) -> Result<Status> {
        let mut status = Status {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Status>() as u32,
            phase: 0,
            result: 0,
            completed_jobs: 0,
        };
        self.check(unsafe { (self.api.status)(self.ptr.as_ptr(), &mut status) })?;
        Ok(status)
    }
    pub fn cancel(&self) -> Result<()> {
        self.check(unsafe { (self.api.cancel)(self.ptr.as_ptr()) })
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        // Native destruction joins the engine before releasing its resources.
        // The process supervisor enforces the external-collective grace timeout.
        unsafe {
            (self.api.cancel)(self.ptr.as_ptr());
            (self.api.destroy)(self.ptr.as_ptr());
        }
    }
}

/// Process APIs are kept at the same audited FFI boundary as CUDA entry points.
pub fn kill_process_group(pid: u32) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        let pid = i32::try_from(pid).map_err(|_| std::io::Error::other("pid overflow"))?;
        if pid <= 1 {
            return Err(std::io::Error::other("invalid worker pid"));
        }
        // Each owned worker starts a new process group with its pid as pgid.
        if unsafe { libc::kill(-pid, libc::SIGKILL) } != 0 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error);
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = pid;
    }
    Ok(())
}

pub fn parent_process_id() -> u32 {
    #[cfg(unix)]
    {
        unsafe { libc::getppid() as u32 }
    }
    #[cfg(not(unix))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn c_layout() {
        assert_eq!(std::mem::size_of::<Job>(), 48);
        assert_eq!(std::mem::size_of::<Status>(), 24);
        assert_eq!(std::mem::offset_of!(Job, state), 16);
        assert_eq!(std::mem::offset_of!(Job, directory), 32);
    }
    #[test]
    fn absent_library_is_an_error_not_a_cuda_import() {
        assert!(NativeLibrary::load(Path::new("/no-such-beam-library.so")).is_err());
        assert!(arguments(&["bad\0arg".into()]).is_err());
    }
}
