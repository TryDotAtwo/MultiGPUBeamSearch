//! All borrowed-pointer conversion and panic containment for host history.
use beam_core::{
    history::{Entry, Stats},
    CandidateMeta,
};
use std::{
    ffi::{c_char, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Mutex,
};

pub trait HistorySink: Send + Sync {
    fn initialize(&self, json: &[u8]) -> Result<(), String>;
    fn write(&self, depth: u32, batch: &[CandidateMeta]) -> Result<(), String>;
    fn read(&self, depth: u32, index: u64) -> Result<Entry, String>;
    fn stats(&self) -> Result<Stats, String>;
    fn close(&self) -> Result<(), String>;
}
#[repr(C)]
pub(crate) struct Api {
    version: u32,
    size: u32,
    data: *mut c_void,
    initialize: unsafe extern "C" fn(*mut c_void, *const c_char, u64) -> i32,
    write: unsafe extern "C" fn(*mut c_void, u32, *const c_void, u64) -> i32,
    read: unsafe extern "C" fn(*mut c_void, u32, u64, *mut Entry) -> i32,
    stats: unsafe extern "C" fn(*mut c_void, *mut Stats) -> i32,
    close: unsafe extern "C" fn(*mut c_void) -> i32,
    error: unsafe extern "C" fn(*mut c_void, *mut c_char, u64, *mut u64) -> i32,
}
pub(crate) struct Bridge {
    sink: Box<dyn HistorySink>,
    error: Mutex<String>,
}
impl Bridge {
    pub(crate) fn new(sink: Box<dyn HistorySink>) -> Self {
        Self {
            sink,
            error: Mutex::new(String::new()),
        }
    }
    pub(crate) fn table(&mut self) -> Api {
        Api {
            version: 1,
            size: std::mem::size_of::<Api>() as u32,
            data: (self as *mut Self).cast(),
            initialize,
            write,
            read,
            stats,
            close,
            error,
        }
    }
}
unsafe fn guarded(
    data: *mut c_void,
    call: impl FnOnce(&dyn HistorySink) -> Result<(), String>,
) -> i32 {
    if data.is_null() {
        return 1;
    }
    // Owner is boxed in Context and native joins all callbacks before drop.
    let bridge = &*data.cast::<Bridge>();
    let result = catch_unwind(AssertUnwindSafe(|| call(bridge.sink.as_ref())));
    let error = match result {
        Ok(Ok(())) => return 0,
        Ok(Err(error)) => error,
        Err(_) => "Rust history callback panicked".into(),
    };
    *bridge.error.lock().unwrap_or_else(|e| e.into_inner()) = error;
    3
}
unsafe extern "C" fn initialize(data: *mut c_void, json: *const c_char, len: u64) -> i32 {
    guarded(data, |sink| {
        if json.is_null() || len == 0 || len > 65536 {
            return Err("invalid history configuration view".into());
        }
        sink.initialize(std::slice::from_raw_parts(json.cast(), len as usize))
    })
}
unsafe extern "C" fn write(data: *mut c_void, depth: u32, ptr: *const c_void, len: u64) -> i32 {
    guarded(data, |sink| {
        if len == 0 {
            return sink.write(depth, &[]);
        }
        if ptr.is_null()
            || (ptr as usize) % std::mem::align_of::<CandidateMeta>() != 0
            || len > (isize::MAX as u64) / 32
        {
            return Err("invalid aligned history batch view".into());
        }
        // Native guarantees D2H completion and exclusive slot ownership until
        // this synchronous callback returns. No device pointer becomes a slice.
        sink.write(depth, std::slice::from_raw_parts(ptr.cast(), len as usize))
    })
}
unsafe extern "C" fn read(data: *mut c_void, depth: u32, index: u64, out: *mut Entry) -> i32 {
    guarded(data, |sink| {
        if out.is_null() {
            return Err("null history entry".into());
        }
        *out = sink.read(depth, index)?;
        Ok(())
    })
}
unsafe extern "C" fn stats(data: *mut c_void, out: *mut Stats) -> i32 {
    guarded(data, |sink| {
        if out.is_null() {
            return Err("null history stats".into());
        }
        *out = sink.stats()?;
        Ok(())
    })
}
unsafe extern "C" fn close(data: *mut c_void) -> i32 {
    guarded(data, |sink| sink.close())
}
unsafe extern "C" fn error(
    data: *mut c_void,
    out: *mut c_char,
    capacity: u64,
    required: *mut u64,
) -> i32 {
    if data.is_null() || required.is_null() {
        return 1;
    }
    let bridge = &*data.cast::<Bridge>();
    let message = bridge.error.lock().unwrap_or_else(|e| e.into_inner());
    *required = message.len() as u64 + 1;
    if capacity < *required {
        return 6;
    }
    if out.is_null() {
        return 1;
    }
    std::ptr::copy_nonoverlapping(message.as_ptr(), out.cast(), message.len());
    *out.add(message.len()) = 0;
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Panics;
    impl HistorySink for Panics {
        fn initialize(&self, _: &[u8]) -> Result<(), String> {
            panic!("test panic")
        }
        fn write(&self, _: u32, batch: &[CandidateMeta]) -> Result<(), String> {
            assert!(batch.is_empty());
            Ok(())
        }
        fn read(&self, _: u32, _: u64) -> Result<Entry, String> {
            Err("read failure".into())
        }
        fn stats(&self) -> Result<Stats, String> {
            Ok(Stats::default())
        }
        fn close(&self) -> Result<(), String> {
            Ok(())
        }
    }
    #[test]
    fn views_empty_alignment_and_panic_boundary() {
        let mut owner = Bridge::new(Box::new(Panics));
        let api = owner.table();
        assert_eq!(std::mem::size_of::<Api>(), 64);
        assert_eq!(std::mem::size_of::<Stats>(), 112);
        unsafe {
            assert_eq!((api.initialize)(api.data, b"{}".as_ptr().cast(), 2), 3);
            let mut needed = 0;
            assert_eq!(
                (api.error)(api.data, std::ptr::null_mut(), 0, &mut needed),
                6
            );
            let mut text = vec![0u8; needed as usize];
            assert_eq!(
                (api.error)(api.data, text.as_mut_ptr().cast(), needed, &mut needed),
                0
            );
            assert!(String::from_utf8(text).unwrap().contains("panicked"));
            assert_eq!((api.write)(api.data, 0, std::ptr::null(), 0), 0);
            assert_eq!(
                (api.write)(api.data, 0, std::ptr::dangling::<u8>().cast(), 1),
                3
            );
            assert_eq!((api.write)(api.data, 0, std::ptr::null(), 1), 3);
        }
    }
}
