use beam_core::{config::NativeOptions, profile_policy::Hardware, search::PermutationGraph};
use beam_runtime::{
    artifacts::Artifact,
    downstream_job::{prepare, Request},
    profile_cache::cache_key,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};
#[test]
fn downstream_identity_binds_quality_graph_geometry_and_semantic_environment() {
    let graph = PermutationGraph {
        center: vec![0, 1, 2],
        generators: vec![vec![1, 0, 2], vec![0, 2, 1]],
    };
    let artifacts = vec![Artifact {
        path: PathBuf::from("/weights"),
        sha256: "a".repeat(64),
        bytes: None,
    }];
    let signature = json!({"world_size":2,"device_indices":[0,1],"gpu_uuids":["a","b"],"max_batch":32,
        "requested_beam_width":65,"graph_contract_sha256":cache_key(&json!(graph)).unwrap(),"artifact_files":artifacts});
    let row = |rank| {
        json!({"batch":16,"parents":33,"device":rank,"seconds":vec![0.01;7],
        "correctness_passed":true,"numeric_error":0,"torch_reserved_peak_bytes":1000})
    };
    let sample = json!([{"index":0,"uuid":"a","temperature":50,"throttled":false},
        {"index":1,"uuid":"b","temperature":50,"throttled":false}]);
    let inference = json!({"signature":signature,"phase":"inference_verified","parent_batch":16,
        "reserve_bytes":1000+(512u64<<20),"records":[row(0),row(1)],"estimate":{"median":9999.},
        "gpu_telemetry":{"16":{"samples":[sample.clone(),sample],"errors":[],"interval_seconds":5.,"throttled":false}}});
    let request = Request {
        options: NativeOptions::default(),
        graph,
        artifacts,
        runner: Artifact {
            path: PathBuf::from("/runner"),
            sha256: "b".repeat(64),
            bytes: None,
        },
        inference,
        environment: BTreeMap::from([("BEAM_SHARD_CAPACITY_SCALE_PPM".into(), "1250000".into())]),
        beam_width: 65,
        row_factor: 2,
        hardware: Hardware {
            sm_count: 28,
            l2_bytes: 3 << 20,
        },
        run_dir: PathBuf::from("/run"),
        cwd: PathBuf::from("/work"),
        physical_bytes: 16,
        puzzle_id: 0,
        grow: true,
        fixtures: vec![],
    };
    let initial = prepare(&request).unwrap();
    assert_eq!(initial.overrides["BEAM_B_MICRO"], "32");
    assert_eq!(initial.tuning.micro, 16);
    assert!((initial.tuning.inference_seconds - 0.01 / 66.).abs() < 1e-12);
    assert_eq!(initial.environment["BEAM_SOLVED_NEIGHBORHOOD_RADIUS"], "0");
    let mut component_only = request.clone();
    component_only.grow = false;
    assert!(
        prepare(&component_only).is_ok(),
        "component probes do not use full-frontier fixture files"
    );
    component_only.options.calibration_full_frontier = true;
    assert!(
        prepare(&component_only).is_err(),
        "full-step fixture mode needs all rank inputs"
    );
    let mut changed = request.clone();
    changed.run_dir = PathBuf::from("/another-run");
    changed
        .environment
        .insert("BEAM_NCCL_RUN_ID".into(), "different".into());
    assert_eq!(initial.key, prepare(&changed).unwrap().key);
    changed
        .environment
        .insert("BEAM_SHARD_CAPACITY_SCALE_PPM".into(), "1500000".into());
    assert_ne!(initial.key, prepare(&changed).unwrap().key);
    for change in 0..5 {
        let mut changed = request.clone();
        match change {
            0 => changed.beam_width = 66,
            1 => changed.inference["records"][0]["numeric_error"] = json!(1),
            2 => changed.graph.center.swap(0, 1),
            3 => changed.artifacts[0].sha256 = "c".repeat(64),
            _ => changed.row_factor = 3,
        }
        assert!(prepare(&changed).is_err(), "{change}");
    }
}
