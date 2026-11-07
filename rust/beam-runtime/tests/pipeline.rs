use beam_runtime::pipeline::{parse_depths, parse_plan};
#[cfg(unix)]
#[test]
fn normal_pipeline_cleanup_does_not_cancel_parent_job() {
    use beam_runtime::pipeline::{Config, PipelineProbe};
    use std::{
        fs,
        os::unix::fs::PermissionsExt,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::Duration,
    };
    let dir = std::env::temp_dir().join(format!("beam-pipeline-owner-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let runner = dir.join("runner");
    fs::write(&runner,"#!/usr/bin/env python\nimport os,sys,json\nprint(json.dumps(dict(ready=True,rank=int(os.environ['RANK']),component_protocol='persistent-exact-capacity-v1')),flush=True)\nfor line in sys.stdin: pass\n").unwrap();
    fs::set_permissions(&runner, fs::Permissions::from_mode(0o700)).unwrap();
    let token = Arc::new(AtomicBool::new(false));
    let config = Config {
        runner,
        environment: std::env::vars().collect(),
        beam: 65,
        world: 2,
        cwd: dir.clone(),
        directory: dir.join("pipeline"),
        fixtures: vec![],
        physical_bytes: 16,
        repeats: 6,
        puzzle_id: 0,
        grow: true,
        gpu_uuids: vec![],
        device_indices: vec![0, 1],
    };
    let mut probe =
        PipelineProbe::launch_cancellable(config, Duration::from_secs(5), token.clone()).unwrap();
    probe.close().unwrap();
    drop(probe);
    assert!(!token.load(Ordering::Acquire));
    fs::remove_dir_all(dir).unwrap();
}
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

#[cfg(unix)]
#[test]
fn native_capacity_diagnostics_survive_coarse_rank_failure() {
    use beam_runtime::{
        calibration::ProfileEvaluator,
        pipeline::{Config, PipelineProbe},
    };
    use serde_json::json;
    use std::{fs, os::unix::fs::PermissionsExt, time::Duration};
    let dir = std::env::temp_dir().join(format!("beam-pipeline-error-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let runner = dir.join("runner");
    fs::write(&runner,"#!/usr/bin/env python\nimport os,sys,json\nif os.environ.get('BEAM_CALIBRATION_PLAN_SESSION')=='1':\n print(json.dumps(dict(ready=True,rank=int(os.environ['RANK']),component_protocol='persistent-exact-capacity-v1')),flush=True)\n for line in sys.stdin: pass\nelse:\n print('production_runner_error=capacity exceeded required=99 available=64',flush=True)\n sys.exit(1)\n").unwrap();
    fs::set_permissions(&runner, fs::Permissions::from_mode(0o700)).unwrap();
    let config = Config {
        runner,
        environment: std::env::vars().collect(),
        beam: 65,
        world: 1,
        cwd: dir.clone(),
        directory: dir.join("pipeline"),
        fixtures: vec![],
        physical_bytes: 16,
        repeats: 6,
        puzzle_id: 0,
        grow: true,
        gpu_uuids: vec![],
        device_indices: vec![0],
    };
    let mut probe = PipelineProbe::launch(config, Duration::from_secs(5)).unwrap();
    probe.finish_planning().unwrap();
    let plans = vec![
        json!({"GLOBAL_BEAM_WIDTH_EFFECTIVE":65,"BEAM_WIDTH_ALIGNMENT":32,"SHARD_COUNT":1,"B_MICRO":16,
        "WORLD_SIZE":1,"STREAM4_BATCH_ALIGNMENT":32,"LOCAL_RANK":0,"estimated_required_device_bytes":500,"gpu_budget_bytes":1000}),
    ];
    let error = probe
        .measure(&std::collections::BTreeMap::new(), &plans)
        .unwrap_err()
        .to_string();
    assert!(error.contains("required=99 available=64"), "{error}");
    probe.close().unwrap();
    fs::remove_dir_all(dir).unwrap();
}
