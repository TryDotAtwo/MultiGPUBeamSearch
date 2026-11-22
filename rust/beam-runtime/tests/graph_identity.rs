use beam_core::search::PermutationGraph;
use beam_runtime::search_session::SessionIdentity;
use serde_json::json;

fn identity(graph: PermutationGraph, graph_hash: &str) -> SessionIdentity {
    SessionIdentity {
        graph,
        graph_hash: graph_hash.into(),
        model_artifact_hash: "b".repeat(64),
        model_backend: "mlp".into(),
        runtime_build: json!({"binary_sha256":"c".repeat(64)}),
        devices: vec![0],
        beam_width: 64,
        max_steps: 1,
        options: json!({}),
        cuda_visible_devices: None,
        cuda_device_order: None,
    }
}

#[test]
fn accepts_python_checkpoint_hashes_at_state_boundaries() {
    // Golden hashes were generated independently with Python's
    // hashlib.sha256(json.dumps(data, sort_keys=True).encode()).
    let examples = [
        (
            1,
            "1c360763a80bfe790e8e4ab2c32c7b85a99ad7a9f98144e5205a0f0c6d340031",
        ),
        (
            2,
            "a0ff37ebeac0a98933a72fd5c08051d2d7fe7e529ef008f5cdd8b3721ce7d0c7",
        ),
        (
            4,
            "fc55344ec7953697b6da8da8899ca8eff36c93a7c375c808f34eb346517c2e68",
        ),
        (
            11,
            "77b620be322d9a2e887655e7277ad2c582facd61dde818e6f9f523e47b9d3329",
        ),
        (
            128,
            "a7880e51c00909275b8fdb08d2d74b01af7313e68f01296ca16f6ebc05fee2b9",
        ),
        (
            129,
            "f47298a22340c6dc4e4199d72983e047c83cedb6aed1f0c8f8233ad83aca0f3a",
        ),
        (
            150,
            "00c42327a2c30c8b5f85045260af915703ef6bcbdbe654093c8da575118ba86e",
        ),
    ];
    for (size, digest) in examples {
        let center = (0..size)
            .map(|i| {
                if size > 128 && i % 5 == 0 {
                    255
                } else if size > 2 {
                    (i % 4) as u8
                } else {
                    i as u8
                }
            })
            .collect();
        let generators = vec![(0..size as u32).collect(), (0..size as u32).rev().collect()];
        identity(PermutationGraph { center, generators }, digest)
            .validate()
            .unwrap();
    }
}

#[test]
fn accepts_python_checkpoint_hash_at_maximum_generator_count() {
    let generators = (0..255)
        .map(|i| (0..4).map(|j| ((j + i) % 4) as u32).collect())
        .collect();
    identity(
        PermutationGraph {
            center: vec![0, 1, 2, 3],
            generators,
        },
        "058813d078d9510f8d5dc1f576687de41dde13467b485ff87329314e61a1d45b",
    )
    .validate()
    .unwrap();
}

#[test]
fn rejects_stale_hash_for_center_or_ordered_generators() {
    let graph = PermutationGraph {
        center: vec![0, 1, 2, 3],
        generators: vec![vec![1, 0, 2, 3], vec![0, 1, 3, 2]],
    };
    let mut job = identity(
        graph,
        "0dfd25a031f018028e013a6cbbf83b3588c61b5b26897bb5321d117d7d3cca8d",
    );
    job.validate().unwrap();
    job.graph.center.swap(0, 1);
    assert!(job
        .validate()
        .unwrap_err()
        .to_string()
        .contains("graph hash"));
    job.graph.center.swap(0, 1);
    job.graph.generators.swap(0, 1);
    assert!(job
        .validate()
        .unwrap_err()
        .to_string()
        .contains("graph hash"));
    job.graph.generators.swap(0, 1);
    job.graph.generators[0].swap(0, 1);
    assert!(job
        .validate()
        .unwrap_err()
        .to_string()
        .contains("graph hash"));
    job.graph.generators[0].swap(0, 1);
    job.graph_hash.replace_range(0..1, "f");
    assert!(job
        .validate()
        .unwrap_err()
        .to_string()
        .contains("graph hash"));
}
