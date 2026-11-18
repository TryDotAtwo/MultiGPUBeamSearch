//! Rust-owned decisions at the two nonterminal permutation-depth boundaries.
pub use beam_native_sys::depth_control::{Checkpoint as DepthCheckpoint, TrackedPathStop};
use beam_native_sys::depth_control::DepthControlSink;

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
}
