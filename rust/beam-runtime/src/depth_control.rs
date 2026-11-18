//! Rust-owned decisions at the two nonterminal permutation-depth boundaries.
pub use beam_native_sys::depth_control::{
    Checkpoint as DepthCheckpoint, CheckpointV2 as DepthCheckpointV2, TrackedPathStop,
};
use beam_native_sys::depth_control::{
    Decision, DepthControlSink, STOP_REASON_CAPACITY, STOP_REASON_DEPTH, STOP_REASON_NONE,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SolveBucketStatus {
    pub current_depth_index: u32,
    pub completed_depths: u32,
    pub max_solutions: u64,
    pub accepted_solutions: u64,
    pub stop_depth: u32,
    pub first_found_depth_index: u32,
    pub extra_depths: u32,
    pub found_any: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveBucketStopReason {
    Continue,
    DepthReached,
    CapacityReached,
}
impl SolveBucketStopReason {
    fn abi_code(self) -> u32 {
        match self {
            Self::Continue => STOP_REASON_NONE,
            Self::DepthReached => STOP_REASON_DEPTH,
            Self::CapacityReached => STOP_REASON_CAPACITY,
        }
    }
}

pub fn solve_bucket_stop_reason(status: SolveBucketStatus) -> SolveBucketStopReason {
    if status.max_solutions != 0 && status.accepted_solutions >= status.max_solutions {
        SolveBucketStopReason::CapacityReached
    } else if status.stop_depth != 0 && status.completed_depths >= status.stop_depth {
        SolveBucketStopReason::DepthReached
    } else if status.stop_depth == 0
        && status.found_any
        && status.current_depth_index
            >= status
                .first_found_depth_index
                .wrapping_add(status.extra_depths)
    {
        SolveBucketStopReason::DepthReached
    } else {
        SolveBucketStopReason::Continue
    }
}

pub fn continue_after(checkpoint: DepthCheckpoint) -> bool {
    match checkpoint {
        DepthCheckpoint::BeforeFrontierCollective {
            calibration_limit_reached,
        } => !calibration_limit_reached,
        DepthCheckpoint::AfterFrontierCollective {
            frontier_live,
            tracked_path_stop,
        } => frontier_live && tracked_path_stop.is_none(),
    }
}

#[derive(Default)]
pub struct NativeDepthController;
impl DepthControlSink for NativeDepthController {
    fn decide(
        &self,
        checkpoint: beam_native_sys::depth_control::Checkpoint,
    ) -> Result<bool, String> {
        Ok(continue_after(checkpoint))
    }

    fn decide_v2(
        &self,
        checkpoint: beam_native_sys::depth_control::CheckpointV2,
    ) -> Result<Decision, String> {
        let stop_reason = match checkpoint {
            DepthCheckpointV2::SolveBucketStop {
                max_solutions,
                accepted_solutions,
                completed_depths,
                current_depth_index,
                stop_depth,
                first_found_depth_index,
                extra_depths,
                found_any,
            } => solve_bucket_stop_reason(SolveBucketStatus {
                current_depth_index,
                completed_depths,
                max_solutions,
                accepted_solutions,
                stop_depth,
                first_found_depth_index,
                extra_depths,
                found_any,
            })
            .abi_code(),
            _ => 0,
        };
        let continue_search = match checkpoint {
            DepthCheckpointV2::BeforeFrontierCollective {
                calibration_limit_reached,
            } => continue_after(DepthCheckpoint::BeforeFrontierCollective {
                calibration_limit_reached,
            }),
            DepthCheckpointV2::AfterFrontierCollective {
                frontier_live,
                tracked_path_stop,
            } => continue_after(DepthCheckpoint::AfterFrontierCollective {
                frontier_live,
                tracked_path_stop,
            }),
            DepthCheckpointV2::SolveBucketStop { .. } => stop_reason == STOP_REASON_NONE,
        };
        Ok(Decision {
            continue_search,
            stop_reason,
        })
    }
}
