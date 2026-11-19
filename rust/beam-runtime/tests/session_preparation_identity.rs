use beam_runtime::search_session::{PreparationIdentity, SearchSession};

const PREPARATION: &str = r#"{"graph_hash":"a","geometry":[2,2,1],"center":[0,1],"generators":[[1,0]],"model_artifact_hash":"b","artifact_sizes":[17],"model_backend":"mlp","devices":[0,1],"binding_scope":false,"beam_width":64,"max_steps":4,"options_repr":"NativeOptions()","cuda_visible_devices":"0,1","cuda_device_order":"PCI_BUS_ID","runtime_build":{"binary_sha256":"c","inference_backend":"cutlass"}}"#;
const REORDERED: &str = r#"{"runtime_build":{"inference_backend":"cutlass","binary_sha256":"c"},"cuda_device_order":"PCI_BUS_ID","cuda_visible_devices":"0,1","options_repr":"NativeOptions()","max_steps":4,"beam_width":64,"binding_scope":false,"devices":[0,1],"model_backend":"mlp","artifact_sizes":[17],"model_artifact_hash":"b","generators":[[1,0]],"center":[0,1],"geometry":[2,2,1],"graph_hash":"a"}"#;

fn identity(raw: &str) -> PreparationIdentity {
    serde_json::from_str(raw).unwrap()
}

#[test]
fn preparation_identity_is_typed_structurally_compared_and_invalidated_by_close() {
    let mut session = SearchSession::new();
    let preparation = identity(PREPARATION);
    let reordered = identity(REORDERED);

    assert!(!session.preparation_matches(&preparation).unwrap());
    session.remember_preparation(&preparation).unwrap();
    assert!(session.preparation_matches(&reordered).unwrap());

    let changed_raw = PREPARATION.replace(
        "\"model_artifact_hash\":\"b\"",
        "\"model_artifact_hash\":\"d\"",
    );
    let changed = identity(&changed_raw);
    assert!(!session.preparation_matches(&changed).unwrap());
    assert!(serde_json::from_str::<PreparationIdentity>("{}").is_err());

    session.close().unwrap();
    assert!(session.preparation_matches(&preparation).is_err());
}
