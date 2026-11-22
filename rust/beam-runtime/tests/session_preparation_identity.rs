use beam_runtime::search_session::{PreparationIdentity, RequestedBeam, SearchSession};

const PREPARATION: &str = r#"{"graph_hash":"a","geometry":[2,2,1],"center":[0,1],"generators":[[1,0]],"model_artifact_hash":"b","artifact_sizes":[17],"model_backend":"mlp","devices":[0,1],"binding_scope":false,"beam_width":64,"max_steps":4,"options":{"inference_backend":"auto"},"resolved_inference_plan":{"backend":"cutlass","policy":"cutlass-default-v1","inference_row_limit":8192},"cuda_visible_devices":"0,1","cuda_device_order":"PCI_BUS_ID","runtime_build":{"binary_sha256":"c","inference_backend":"cutlass"}}"#;
const REORDERED: &str = r#"{"runtime_build":{"inference_backend":"cutlass","binary_sha256":"c"},"cuda_device_order":"PCI_BUS_ID","cuda_visible_devices":"0,1","options":{"inference_backend":"auto"},"resolved_inference_plan":{"backend":"cutlass","policy":"cutlass-default-v1","inference_row_limit":8192},"max_steps":4,"beam_width":64,"binding_scope":false,"devices":[0,1],"model_backend":"mlp","artifact_sizes":[17],"model_artifact_hash":"b","generators":[[1,0]],"center":[0,1],"geometry":[2,2,1],"graph_hash":"a"}"#;

fn identity(raw: &str) -> PreparationIdentity {
    serde_json::from_str(raw).unwrap()
}

#[test]
fn requested_beam_round_trips_auto_max_and_legacy_numeric_json() {
    let automatic_raw = PREPARATION.replace("\"beam_width\":64", "\"beam_width\":\"max\"");
    let automatic = identity(&automatic_raw);
    assert_eq!(automatic.requested_beam, RequestedBeam::AutoMax);
    let automatic_json = serde_json::to_value(&automatic).unwrap();
    assert_eq!(automatic_json["beam_width"].as_str(), Some("max"));
    assert!(automatic_json.get("requested_beam").is_none());

    let exact = identity(PREPARATION);
    assert_eq!(exact.requested_beam, RequestedBeam::Exact(64));
    let exact_json = serde_json::to_value(&exact).unwrap();
    assert_eq!(exact_json["beam_width"].as_u64(), Some(64));
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

#[test]
fn begin_preparation_reuses_matches_and_invalidates_changes() {
    let mut session = SearchSession::new();
    let preparation = identity(PREPARATION);
    let reordered = identity(REORDERED);
    let mut changed_plan = identity(PREPARATION);
    changed_plan
        .resolved_inference_plan
        .as_mut()
        .unwrap()
        .inference_row_limit = 4096;
    let mut changed_model = identity(PREPARATION);
    changed_model.model_artifact_hash = "d".into();

    assert!(!session.begin_preparation(&preparation).unwrap());
    session.remember_preparation(&preparation).unwrap();
    assert!(session.begin_preparation(&reordered).unwrap());
    assert!(!session.begin_preparation(&changed_plan).unwrap());
    session.remember_preparation(&preparation).unwrap();
    assert!(!session.begin_preparation(&changed_model).unwrap());
    assert!(!session.preparation_matches(&preparation).unwrap());

    session.close().unwrap();
    assert!(session.begin_preparation(&preparation).is_err());
}
