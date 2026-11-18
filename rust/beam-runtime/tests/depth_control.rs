use beam_runtime::depth_control::{continue_after, DepthCheckpoint, TrackedPathStop};

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
