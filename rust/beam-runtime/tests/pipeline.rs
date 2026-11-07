use beam_runtime::pipeline::{parse_depths, parse_plan};
#[test]
fn complete_depth_parser_rejects_missing_reordered_and_changed_workloads() {
    assert_eq!(parse_depths("noise\ncalibration_depth=0 parents=7 depth_sec=1.5\ncalibration_depth=1 parents=7 depth_sec=2\n",7,2).unwrap(),vec![1.5,2.]);
    for text in [
        "calibration_depth=1 parents=7 depth_sec=1",
        "calibration_depth=0 parents=6 depth_sec=1",
        "calibration_depth=0 parents=7 depth_sec=NaN",
    ] {
        assert!(parse_depths(text, 7, 1).is_err());
    }
    assert!(parse_depths("calibration_depth=0 parents=7 depth_sec=1", 7, 2).is_err());
    assert!(parse_plan("GLOBAL_BEAM_WIDTH_EFFECTIVE=5\nGLOBAL_BEAM_WIDTH_EFFECTIVE=6").is_err());
    assert_eq!(
        parse_plan("GLOBAL_BEAM_WIDTH_EFFECTIVE=5\nB_MICRO=64").unwrap()["B_MICRO"],
        64
    );
}
