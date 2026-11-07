use beam_core::{config::NativeOptions, search::PermutationGraph};
use beam_runtime::{
    artifacts::Artifact,
    inference::Protocol,
    inference_job::{prepare, Request},
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

#[test]
fn request_owns_bounds_cache_identity_and_legal_samples() {
    let request = Request {
        options: NativeOptions {
            calibration_max_batch: 32,
            ..NativeOptions::default()
        },
        graph: PermutationGraph {
            center: vec![0, 1, 2],
            generators: vec![vec![1, 0, 2], vec![0, 2, 1]],
        },
        helper: Artifact {
            path: PathBuf::from("/helper"),
            sha256: "a".repeat(64),
            bytes: None,
        },
        artifacts: vec![Artifact {
            path: PathBuf::from("/weights/model"),
            sha256: "b".repeat(64),
            bytes: None,
        }],
        probe_manifest: json!({"weights_dir":"/weights"}),
        ensemble: false,
        signature: json!({"world_size":2,"device_indices":[0,1],"gpu_uuids":["a","b"]}),
        beam_width: 65,
        protocol: Protocol::Oneshot,
        capacity_bootstrap: false,
        run_dir: PathBuf::from("/run"),
        environment: BTreeMap::new(),
        shared_profile: None,
    };
    let first = prepare(&request).unwrap();
    assert_eq!(
        (first.capacity, first.cap, first.parents, first.world),
        (32, 32, 33, 2)
    );
    assert_eq!(first.samples.len(), 128);
    for state in &first.samples {
        request.graph.validate_start(state).unwrap();
    }
    assert_eq!(first.samples, prepare(&request).unwrap().samples);
    assert_eq!(
        first.signature["search_policy"],
        "rust-calibration-owner-v1"
    );
    let mut changed = request.clone();
    changed.options.calibration_max_batch = 16;
    assert_ne!(first.key, prepare(&changed).unwrap().key);
    changed = request.clone();
    changed.helper.sha256 = "c".repeat(64);
    assert_ne!(first.key, prepare(&changed).unwrap().key);
    changed = request.clone();
    changed.beam_width = u64::MAX;
    assert_eq!(prepare(&changed).unwrap().parents, 8192);
    changed = request.clone();
    changed.signature["device_indices"] = json!([0, 0]);
    assert!(prepare(&changed).is_err());
    changed = request.clone();
    changed.artifacts.clear();
    assert!(prepare(&changed).is_err());
    changed = request.clone();
    changed.beam_width = 0;
    assert!(prepare(&changed).is_err());
}
