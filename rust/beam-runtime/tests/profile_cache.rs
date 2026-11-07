use beam_runtime::artifacts::{sha256_file, Artifact};
use beam_runtime::profile_cache::{cache_key, read_inference, write_inference};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs};

fn fixture() -> (Value, Value, BTreeMap<String, u64>) {
    let signature = json!({"schema":10,"search_policy":"rust-inference-v1",
        "world_size":2,"device_indices":[0,1],"gpu_uuids":["a","b"],"max_batch":32});
    let row = |rank| {
        json!({"batch":16,"parents":129,"device":rank,"seconds":vec![0.1;7],
        "correctness_passed":true,"numeric_error":0,"torch_reserved_peak_bytes":1000})
    };
    let sample = json!([{"index":0,"uuid":"GPU-a","temperature":50,"throttled":false},
        {"index":1,"uuid":"GPU-b","temperature":51,"throttled":false}]);
    let data = json!({"signature":signature,"phase":"inference_verified","parent_batch":16,
        "reserve_bytes":1000+(512u64<<20),"records":[row(0),row(1)],
        "gpu_telemetry":{"16":{"samples":[sample.clone(),sample],"errors":[],
            "interval_seconds":5.,"throttled":false}},
        "capacity_limited_search":true,"initial_free_memory_mib":{"a":8000,"b":8001},
        "estimate":{"median":9999.},"cache_hit":false});
    (
        signature,
        data,
        BTreeMap::from([("a".into(), 8000), ("b".into(), 8001)]),
    )
}

#[test]
fn native_timing_receipts_keep_float_bits_and_content_keys_across_json_cache() {
    // Real GPU receipt values that drifted one ULP under the default parser,
    // invalidating a full-step cache on every otherwise compatible read.
    let values = [0.014777312278747559f64, 0.009699328422546387f64];
    let original = json!({"timings":values});
    let restored: Value = serde_json::from_slice(&serde_json::to_vec(&original).unwrap()).unwrap();
    for (index, expected) in values.iter().enumerate() {
        assert_eq!(
            expected.to_bits(),
            restored["timings"][index].as_f64().unwrap().to_bits()
        );
    }
    assert_eq!(cache_key(&original).unwrap(), cache_key(&restored).unwrap());
}

#[test]
fn artifacts_hash_bound_and_cache_requires_complete_evidence() {
    let root = std::env::temp_dir().join(format!("beam-cache-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let model = root.join("model.bin");
    fs::write(&model, b"abc").unwrap();
    let digest = sha256_file(&model).unwrap();
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    let artifact = Artifact {
        path: model.clone(),
        sha256: digest,
        bytes: Some(3),
    };
    artifact.verify().unwrap();
    fs::write(&model, b"abd").unwrap();
    assert!(artifact.verify().is_err());
    let (signature, data, free) = fixture();
    let path = root.join("profile.json");
    write_inference(&path, &data, &signature, 32).unwrap();
    let hit = read_inference(&path, &signature, 32, Some(&free))
        .unwrap()
        .unwrap();
    assert_eq!(hit["cache_hit"], true);
    assert!((hit["estimate"]["median"].as_f64().unwrap() - 0.1 / 258.).abs() < 1e-12);
    let mut lifted = free.clone();
    lifted.insert("a".into(), 8033);
    assert!(read_inference(&path, &signature, 32, Some(&lifted))
        .unwrap()
        .is_none());
    assert!(read_inference(&path, &signature, 32, None)
        .unwrap()
        .is_none());
    assert!(
        read_inference(&path, &json!({"other":"identity"}), 32, Some(&free))
            .unwrap()
            .is_none()
    );
    for mutate in 0..5 {
        let mut bad = data.clone();
        match mutate {
            0 => bad["records"][1]["device"] = json!(0),
            1 => bad["records"][0]["numeric_error"] = json!(1),
            2 => bad["gpu_telemetry"]["16"]["samples"][1][0]["throttled"] = json!(true),
            3 => bad["reserve_bytes"] = json!(1),
            _ => bad["records"][0]["seconds"] = json!([0.1, 0.1]),
        }
        fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(
            read_inference(&path, &signature, 32, Some(&free))
                .unwrap()
                .is_none(),
            "{mutate}"
        );
        assert!(write_inference(&path, &bad, &signature, 32).is_err());
    }
    fs::write(&path, b"{broken").unwrap();
    assert!(read_inference(&path, &signature, 32, Some(&free))
        .unwrap()
        .is_none());
    assert_eq!(
        cache_key(&json!({"b":2,"a":1})).unwrap(),
        cache_key(&json!({"a":1,"b":2})).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}
