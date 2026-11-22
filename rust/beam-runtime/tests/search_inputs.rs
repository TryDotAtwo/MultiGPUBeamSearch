use beam_core::search::{build_public_graph_contract, PublicGraphContractRequest};

#[test]
fn legacy_runner_inputs_keep_generator_order_and_csv_state() {
    let root = std::env::temp_dir().join(format!(
        "beam-search-inputs-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let request = PublicGraphContractRequest {
        center: vec!["0".into(), "1".into()],
        start: vec!["1".into(), "0".into()],
        generators: vec![vec!["1".into(), "0".into()]; 12],
        generator_name_count: 12,
    };
    let contract = build_public_graph_contract(request).unwrap();

    beam_runtime::search_inputs::write_legacy_search_inputs(&contract, &root).unwrap();

    assert_eq!(
        std::fs::read_to_string(root.join("puzzle_info.json")).unwrap(),
        "{\"central_state\": [0, 1], \"generators\": {\"m0\": [1, 0], \"m1\": [1, 0], \"m2\": [1, 0], \"m3\": [1, 0], \"m4\": [1, 0], \"m5\": [1, 0], \"m6\": [1, 0], \"m7\": [1, 0], \"m8\": [1, 0], \"m9\": [1, 0], \"m10\": [1, 0], \"m11\": [1, 0]}}\n"
    );
    assert_eq!(
        std::fs::read_to_string(root.join("test.csv")).unwrap(),
        "initial_state_id,initial_state\n0,\"1,0\"\n"
    );
    std::fs::remove_dir_all(root).unwrap();
}
