use beam_native_sys::depth_control::{DepthControlSink, STOP_REASON_CAPACITY};
use beam_runtime::depth_control::{
    continue_after, solve_bucket_stop_reason, DepthCheckpoint, DepthCheckpointV2,
    NativeDepthController, SolveBucketStatus, SolveBucketStopReason, TrackedPathStop,
};

#[test]
fn calibration_limit_is_checked_before_the_frontier_collective() {
    assert!(!continue_after(DepthCheckpoint::BeforeFrontierCollective {
        calibration_limit_reached: true,
    }));
    assert!(continue_after(DepthCheckpoint::BeforeFrontierCollective {
        calibration_limit_reached: false,
    }));
}

#[test]
fn frontier_and_tracked_path_stops_keep_the_existing_precedence() {
    assert!(!continue_after(DepthCheckpoint::AfterFrontierCollective {
        frontier_live: false,
        tracked_path_stop: None,
    }));
    assert!(!continue_after(DepthCheckpoint::AfterFrontierCollective {
        frontier_live: false,
        tracked_path_stop: Some(TrackedPathStop::MissingPathWindowElapsed),
    }));
    assert!(!continue_after(DepthCheckpoint::AfterFrontierCollective {
        frontier_live: true,
        tracked_path_stop: Some(TrackedPathStop::MissingPathWindowElapsed),
    }));
    assert!(!continue_after(DepthCheckpoint::AfterFrontierCollective {
        frontier_live: true,
        tracked_path_stop: Some(TrackedPathStop::ExpectedPathExhausted),
    }));
    assert!(continue_after(DepthCheckpoint::AfterFrontierCollective {
        frontier_live: true,
        tracked_path_stop: None,
    }));
}

fn bucket_status() -> SolveBucketStatus {
    SolveBucketStatus {
        current_depth_index: 4,
        completed_depths: 5,
        max_solutions: 0,
        accepted_solutions: 0,
        stop_depth: 0,
        first_found_depth_index: 4,
        extra_depths: 2,
        found_any: false,
    }
}

#[test]
fn solve_bucket_stop_reason_preserves_capacity_depth_and_window_priority() {
    let mut status = bucket_status();
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::Continue
    );

    status.found_any = true;
    status.current_depth_index = 5;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::Continue
    );
    status.current_depth_index = 6;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::DepthReached
    );

    status.stop_depth = 9;
    status.completed_depths = 9;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::DepthReached
    );

    status.max_solutions = 3;
    status.accepted_solutions = 3;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::CapacityReached
    );
}

#[test]
fn solve_bucket_legacy_window_waits_for_a_solution_and_keeps_u32_wraparound() {
    let mut status = bucket_status();
    status.extra_depths = 0;
    status.current_depth_index = 4;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::Continue
    );

    status.found_any = true;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::DepthReached
    );

    status.first_found_depth_index = u32::MAX;
    status.current_depth_index = 0;
    status.extra_depths = 1;
    assert_eq!(
        solve_bucket_stop_reason(status),
        SolveBucketStopReason::DepthReached
    );
}

#[test]
fn native_controller_exports_the_same_solve_bucket_reason_through_abi2() {
    let decision = NativeDepthController
        .decide_v2(DepthCheckpointV2::SolveBucketStop {
            max_solutions: 3,
            accepted_solutions: 3,
            completed_depths: 1,
            current_depth_index: 0,
            stop_depth: 0,
            first_found_depth_index: 0,
            extra_depths: 0,
            found_any: true,
        })
        .expect("valid solve-bucket state");
    assert!(!decision.continue_search);
    assert_eq!(decision.stop_reason, STOP_REASON_CAPACITY);
}
