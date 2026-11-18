//! Per-depth decisions that run on the native engine thread without exposing
//! device pointers or moving CUDA/NCCL operations across the FFI boundary.
use std::{
    ffi::{c_char, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Mutex,
};

pub const ABI_VERSION: u32 = 1;
pub const BEFORE_FRONTIER_COLLECTIVE: u32 = 1;
pub const AFTER_FRONTIER_COLLECTIVE: u32 = 2;
const OK: i32 = 0;
const FAILED: i32 = 3;
const BUFFER_TOO_SMALL: i32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackedPathStop {
    MissingPathWindowElapsed,
    ExpectedPathExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Checkpoint {
    BeforeFrontierCollective {
        calibration_limit_reached: bool,
    },
    AfterFrontierCollective {
        frontier_live: bool,
        tracked_path_stop: Option<TrackedPathStop>,
    },
}

pub trait DepthControlSink: Send + Sync {
    fn decide(&self, checkpoint: Checkpoint) -> Result<bool, String>;
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    pub abi_version: u32,
    pub struct_size: u32,
    pub checkpoint: u32,
    pub calibration_limit_reached: u32,
    pub frontier_live: u32,
    pub tracked_path_stop: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Output {
    pub abi_version: u32,
    pub struct_size: u32,
    pub continue_search: u32,
}

pub type DecideFn = unsafe extern "C" fn(*mut c_void, *const Input, *mut Output) -> i32;
pub type ErrorFn = unsafe extern "C" fn(*mut c_void, *mut c_char, u64, *mut u64) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Api {
    pub abi_version: u32,
    pub struct_size: u32,
    pub user_data: *mut c_void,
    pub decide: DecideFn,
    pub error: ErrorFn,
}

pub struct Bridge {
    sink: Box<dyn DepthControlSink>,
    error: Mutex<String>,
}
impl Bridge {
    pub fn new(sink: Box<dyn DepthControlSink>) -> Self {
        Self {
            sink,
            error: Mutex::new(String::new()),
        }
    }
    pub fn table(&mut self) -> Api {
        Api {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Api>() as u32,
            user_data: (self as *mut Self).cast(),
            decide,
            error,
        }
    }
}

unsafe fn guarded(
    data: *mut c_void,
    call: impl FnOnce(&dyn DepthControlSink) -> Result<(), String>,
) -> i32 {
    if data.is_null() {
        return FAILED;
    }
    // The owning Box stays alive until native destruction joins the engine.
    let bridge = &*data.cast::<Bridge>();
    match catch_unwind(AssertUnwindSafe(|| call(bridge.sink.as_ref()))) {
        Ok(Ok(())) => OK,
        Ok(Err(message)) => {
            *bridge.error.lock().unwrap_or_else(|e| e.into_inner()) = message;
            FAILED
        }
        Err(_) => {
            *bridge.error.lock().unwrap_or_else(|e| e.into_inner()) =
                "Rust depth-control callback panicked".into();
            FAILED
        }
    }
}

fn observation(input: &Input) -> Result<Checkpoint, String> {
    if input.abi_version != ABI_VERSION || input.struct_size < std::mem::size_of::<Input>() as u32 {
        return Err("invalid depth-control input ABI".into());
    }
    match input.checkpoint {
        BEFORE_FRONTIER_COLLECTIVE
            if input.calibration_limit_reached <= 1
                && input.frontier_live == 0
                && input.tracked_path_stop == 0 =>
        {
            Ok(Checkpoint::BeforeFrontierCollective {
                calibration_limit_reached: input.calibration_limit_reached != 0,
            })
        }
        AFTER_FRONTIER_COLLECTIVE
            if input.calibration_limit_reached == 0
                && input.frontier_live <= 1
                && input.tracked_path_stop <= 2 =>
        {
            let tracked_path_stop = match input.tracked_path_stop {
                0 => None,
                1 => Some(TrackedPathStop::MissingPathWindowElapsed),
                2 => Some(TrackedPathStop::ExpectedPathExhausted),
                _ => unreachable!("guarded by the match arm"),
            };
            Ok(Checkpoint::AfterFrontierCollective {
                frontier_live: input.frontier_live != 0,
                tracked_path_stop,
            })
        }
        _ => Err("invalid depth-control checkpoint values".into()),
    }
}

unsafe extern "C" fn decide(data: *mut c_void, input: *const Input, out: *mut Output) -> i32 {
    guarded(data, |sink| {
        if input.is_null() || out.is_null() {
            return Err("null depth-control descriptor".into());
        }
        let checkpoint = observation(&*input)?;
        if (*out).abi_version != ABI_VERSION
            || (*out).struct_size < std::mem::size_of::<Output>() as u32
        {
            return Err("invalid depth-control output ABI".into());
        }
        (*out).continue_search = if sink.decide(checkpoint)? { 1 } else { 0 };
        Ok(())
    })
}

unsafe extern "C" fn error(
    data: *mut c_void,
    out: *mut c_char,
    capacity: u64,
    required: *mut u64,
) -> i32 {
    if data.is_null() || required.is_null() {
        return FAILED;
    }
    let bridge = &*data.cast::<Bridge>();
    let message = bridge.error.lock().unwrap_or_else(|e| e.into_inner());
    *required = message.len() as u64 + 1;
    if capacity < *required {
        return BUFFER_TOO_SMALL;
    }
    if out.is_null() {
        return FAILED;
    }
    std::ptr::copy_nonoverlapping(message.as_ptr(), out.cast(), message.len());
    *out.add(message.len()) = 0;
    OK
}
