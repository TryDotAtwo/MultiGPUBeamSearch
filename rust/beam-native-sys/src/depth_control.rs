//! Per-depth decisions that run on the native engine thread without exposing
//! device pointers or moving CUDA/NCCL operations across the FFI boundary.
use std::{
    ffi::{c_char, c_void},
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Mutex,
};

pub const ABI_VERSION: u32 = 1;
pub const ABI_VERSION_V2: u32 = 2;
pub const BEFORE_FRONTIER_COLLECTIVE: u32 = 1;
pub const AFTER_FRONTIER_COLLECTIVE: u32 = 2;
pub const SOLVE_BUCKET_STOP: u32 = 3;
pub const STOP_REASON_NONE: u32 = 0;
pub const STOP_REASON_DEPTH: u32 = 1;
pub const STOP_REASON_CAPACITY: u32 = 2;
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

/// ABI-2 checkpoints use a separate enum so adding new native observations
/// does not break exhaustive matches against the original public ABI-1 type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointV2 {
    BeforeFrontierCollective {
        calibration_limit_reached: bool,
    },
    AfterFrontierCollective {
        frontier_live: bool,
        tracked_path_stop: Option<TrackedPathStop>,
    },
    SolveBucketStop {
        max_solutions: u64,
        accepted_solutions: u64,
        completed_depths: u32,
        current_depth_index: u32,
        stop_depth: u32,
        first_found_depth_index: u32,
        extra_depths: u32,
        found_any: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Decision {
    pub continue_search: bool,
    pub stop_reason: u32,
}

pub trait DepthControlSink: Send + Sync {
    /// ABI-1 public contract retained for existing Rust implementations.
    fn decide(&self, checkpoint: Checkpoint) -> Result<bool, String>;

    /// ABI-2 extension. Existing sinks keep working through this default;
    /// implementations with solve-bucket policy can return the typed reason.
    fn decide_v2(&self, checkpoint: CheckpointV2) -> Result<Decision, String> {
        match checkpoint {
            CheckpointV2::BeforeFrontierCollective {
                calibration_limit_reached,
            } => Ok(Decision {
                continue_search: self.decide(Checkpoint::BeforeFrontierCollective {
                    calibration_limit_reached,
                })?,
                stop_reason: STOP_REASON_NONE,
            }),
            CheckpointV2::AfterFrontierCollective {
                frontier_live,
                tracked_path_stop,
            } => Ok(Decision {
                continue_search: self.decide(Checkpoint::AfterFrontierCollective {
                    frontier_live,
                    tracked_path_stop,
                })?,
                stop_reason: STOP_REASON_NONE,
            }),
            CheckpointV2::SolveBucketStop {
                max_solutions,
                accepted_solutions,
                completed_depths,
                current_depth_index,
                stop_depth,
                first_found_depth_index,
                extra_depths,
                found_any,
            } => {
                let stop_reason = if max_solutions != 0 && accepted_solutions >= max_solutions {
                    STOP_REASON_CAPACITY
                } else if stop_depth != 0 && completed_depths >= stop_depth {
                    STOP_REASON_DEPTH
                } else if stop_depth == 0
                    && found_any
                    && current_depth_index >= first_found_depth_index.wrapping_add(extra_depths)
                {
                    STOP_REASON_DEPTH
                } else {
                    STOP_REASON_NONE
                };
                Ok(Decision {
                    continue_search: stop_reason == STOP_REASON_NONE,
                    stop_reason,
                })
            }
        }
    }
}

/// ABI-1 layouts are frozen: these are the original 24-byte input and
/// 12-byte output. New checkpoints must use the separately versioned ABI-2.
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

/// ABI-2 adds scalar solve-bucket state while leaving every ABI-1 layout and
/// symbol unchanged.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct InputV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub checkpoint: u32,
    pub calibration_limit_reached: u32,
    pub frontier_live: u32,
    pub tracked_path_stop: u32,
    pub max_solutions: u64,
    pub accepted_solutions: u64,
    pub completed_depths: u32,
    pub current_depth_index: u32,
    pub stop_depth: u32,
    pub first_found_depth_index: u32,
    pub extra_depths: u32,
    pub solve_bucket_found_any: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct OutputV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub continue_search: u32,
    pub stop_reason: u32,
}

pub type DecideFnV2 = unsafe extern "C" fn(*mut c_void, *const InputV2, *mut OutputV2) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ApiV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub user_data: *mut c_void,
    pub decide: DecideFnV2,
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
            decide: decide_v1,
            error,
        }
    }
    pub fn table_v2(&mut self) -> ApiV2 {
        ApiV2 {
            abi_version: ABI_VERSION_V2,
            struct_size: std::mem::size_of::<ApiV2>() as u32,
            user_data: (self as *mut Self).cast(),
            decide: decide_v2,
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

fn observation_v1(input: &Input) -> Result<Checkpoint, String> {
    if input.abi_version != ABI_VERSION || input.struct_size < std::mem::size_of::<Input>() as u32 {
        return Err("invalid ABI-1 depth-control input".into());
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
        _ => Err("invalid ABI-1 depth-control checkpoint values".into()),
    }
}

fn bucket_fields_are_zero(input: &InputV2) -> bool {
    input.max_solutions == 0
        && input.accepted_solutions == 0
        && input.completed_depths == 0
        && input.current_depth_index == 0
        && input.stop_depth == 0
        && input.first_found_depth_index == 0
        && input.extra_depths == 0
        && input.solve_bucket_found_any == 0
}

fn observation_v2(input: &InputV2) -> Result<CheckpointV2, String> {
    if input.abi_version != ABI_VERSION_V2
        || input.struct_size < std::mem::size_of::<InputV2>() as u32
    {
        return Err("invalid ABI-2 depth-control input".into());
    }
    match input.checkpoint {
        BEFORE_FRONTIER_COLLECTIVE
            if input.calibration_limit_reached <= 1
                && input.frontier_live == 0
                && input.tracked_path_stop == 0
                && bucket_fields_are_zero(input) =>
        {
            Ok(CheckpointV2::BeforeFrontierCollective {
                calibration_limit_reached: input.calibration_limit_reached != 0,
            })
        }
        AFTER_FRONTIER_COLLECTIVE
            if input.calibration_limit_reached == 0
                && input.frontier_live <= 1
                && input.tracked_path_stop <= 2
                && bucket_fields_are_zero(input) =>
        {
            let tracked_path_stop = match input.tracked_path_stop {
                0 => None,
                1 => Some(TrackedPathStop::MissingPathWindowElapsed),
                2 => Some(TrackedPathStop::ExpectedPathExhausted),
                _ => unreachable!("guarded by the match arm"),
            };
            Ok(CheckpointV2::AfterFrontierCollective {
                frontier_live: input.frontier_live != 0,
                tracked_path_stop,
            })
        }
        SOLVE_BUCKET_STOP
            if input.calibration_limit_reached == 0
                && input.frontier_live == 0
                && input.tracked_path_stop == 0
                && input.solve_bucket_found_any <= 1 =>
        {
            Ok(CheckpointV2::SolveBucketStop {
                max_solutions: input.max_solutions,
                accepted_solutions: input.accepted_solutions,
                completed_depths: input.completed_depths,
                current_depth_index: input.current_depth_index,
                stop_depth: input.stop_depth,
                first_found_depth_index: input.first_found_depth_index,
                extra_depths: input.extra_depths,
                found_any: input.solve_bucket_found_any != 0,
            })
        }
        _ => Err("invalid ABI-2 depth-control checkpoint values".into()),
    }
}

unsafe extern "C" fn decide_v1(data: *mut c_void, input: *const Input, out: *mut Output) -> i32 {
    guarded(data, |sink| {
        if input.is_null() || out.is_null() {
            return Err("null ABI-1 depth-control descriptor".into());
        }
        let checkpoint = observation_v1(&*input)?;
        if (*out).abi_version != ABI_VERSION
            || (*out).struct_size < std::mem::size_of::<Output>() as u32
        {
            return Err("invalid ABI-1 depth-control output".into());
        }
        (*out).continue_search = if sink.decide(checkpoint)? { 1 } else { 0 };
        Ok(())
    })
}

unsafe extern "C" fn decide_v2(
    data: *mut c_void,
    input: *const InputV2,
    out: *mut OutputV2,
) -> i32 {
    guarded(data, |sink| {
        if input.is_null() || out.is_null() {
            return Err("null ABI-2 depth-control descriptor".into());
        }
        let checkpoint = observation_v2(&*input)?;
        if (*out).abi_version != ABI_VERSION_V2
            || (*out).struct_size < std::mem::size_of::<OutputV2>() as u32
        {
            return Err("invalid ABI-2 depth-control output".into());
        }
        let decision = sink.decide_v2(checkpoint)?;
        (*out).continue_search = if decision.continue_search { 1 } else { 0 };
        (*out).stop_reason = decision.stop_reason;
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
