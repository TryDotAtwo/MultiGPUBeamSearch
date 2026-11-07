use beam_core::config::NativeOptions;
use serde_json::json;

#[test]
fn public_options_roundtrip_and_validation() {
    let mut options = NativeOptions::default();
    options.validate().unwrap();
    options.devices = Some(vec![1, 3]);
    let value = serde_json::to_value(&options).unwrap();
    assert_eq!(value["inference_backend"], "auto");
    assert_eq!(value["solution_mode"], "first");
    assert_eq!(value["calibration_max_batch"], 8192);
    let decoded: NativeOptions = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(decoded, options);
    for (field, invalid) in [
        ("calibration_max_batch", json!(0)),
        ("touch_bfs_radius", json!(13)),
        ("build_jobs", json!(0)),
        ("timeout_seconds", json!(-1.)),
        ("devices", json!([1, 1])),
        ("max_collected_solutions", json!(1_000_001)),
    ] {
        let mut bad = value.clone();
        bad[field] = invalid;
        let decoded: NativeOptions = serde_json::from_value(bad).unwrap();
        assert!(decoded.validate().is_err(), "{field}");
    }
    options.num_gpus = Some(2);
    assert!(options.validate().is_err());
    options.num_gpus = None;
    options.autotune = false;
    options.calibration_discover_capacity = true;
    assert!(options.validate().is_err());
    options.timeout_seconds = f64::INFINITY;
    assert!(options.validate().is_err());
    for (field, invalid) in [
        ("autotune", json!(1)),
        ("build_jobs", json!(true)),
        ("touch_bfs_max_entries", json!(-1)),
        ("unexpected", json!(1)),
    ] {
        let mut bad = value.clone();
        bad[field] = invalid;
        assert!(
            serde_json::from_value::<NativeOptions>(bad).is_err(),
            "{field}"
        );
    }
}
