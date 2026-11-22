use beam_runtime::inference_backend::{resolve, resolve_plan, Backend, DeviceObservation, Request};

fn device(sm: u32, name: &str) -> DeviceObservation {
    DeviceObservation {
        sm,
        name: name.into(),
    }
}

fn request(
    requested: &str,
    model_backend: &str,
    dtype: Option<&str>,
    output_dim: u64,
    devices: Vec<DeviceObservation>,
) -> Request {
    Request {
        requested: requested.into(),
        model_backend: model_backend.into(),
        dtype: dtype.map(str::to_owned),
        output_dim,
        devices,
    }
}

#[test]
fn auto_selection_preserves_t4_ensemble_and_mixed_device_routes() {
    assert_eq!(
        resolve(&request("auto", "mlp", Some("fp16"), 8, vec![device(86, "NVIDIA RTX 3060")]))
            .unwrap(),
        Backend::Cutlass
    );
    assert_eq!(
        resolve(&request("auto", "mlp", Some("fp16"), 8, vec![device(75, "Tesla T4")]))
            .unwrap(),
        Backend::LibTorch
    );
    assert_eq!(
        resolve(&request(
            "auto",
            "mlp",
            Some("fp16"),
            8,
            vec![device(75, "Tesla T4"), device(86, "NVIDIA RTX 3060")]
        ))
        .unwrap(),
        Backend::LibTorch
    );
    assert_eq!(
        resolve(&request(
            "auto",
            "ensemble",
            Some("fp16"),
            64,
            vec![device(75, "NVIDIA RTX A6000")]
        ))
        .unwrap(),
        Backend::LibTorch
    );
    assert_eq!(
        resolve(&request(
            "auto",
            "ensemble",
            Some("fp16"),
            64,
            vec![device(80, "NVIDIA A100")]
        ))
        .unwrap(),
        Backend::Cutlass
    );
}

#[test]
fn explicit_backends_and_scalar_cutlass_head_are_preserved() {
    assert_eq!(
        resolve(&request(
            "libtorch",
            "mlp",
            Some("fp16"),
            7,
            vec![device(86, "NVIDIA A100")]
        ))
        .unwrap(),
        Backend::LibTorch
    );
    assert_eq!(
        resolve(&request(
            "cutlass",
            "mlp",
            Some("fp16"),
            1,
            vec![device(75, "Tesla T4")]
        ))
        .unwrap(),
        Backend::Cutlass
    );
}

#[test]
fn dtype_capability_errors_match_the_public_contract() {
    assert!(resolve(&request(
        "auto",
        "mlp",
        Some("bf16"),
        8,
        vec![device(79, "NVIDIA fixture")]
    ))
    .unwrap_err()
    .to_string()
    .contains("native BF16 Stream1 requires SM80+"));
    assert!(resolve(&request(
        "auto",
        "mlp",
        Some("fp16"),
        8,
        vec![device(74, "NVIDIA fixture")]
    ))
    .unwrap_err()
    .to_string()
    .contains("this native FP16 MLP backend uses SM75+"));
}

#[test]
fn ensemble_and_cutlass_q_head_capabilities_are_checked() {
    assert!(resolve(&request(
        "cutlass",
        "ensemble",
        Some("fp16"),
        64,
        vec![device(75, "NVIDIA RTX A6000")]
    ))
    .unwrap_err()
    .to_string()
    .contains("ensemble CUTLASS requires SM80+"));
    for backend in ["auto", "cutlass"] {
        assert!(resolve(&request(
            backend,
            "mlp",
            Some("fp16"),
            7,
            vec![device(86, "NVIDIA A100")]
        ))
        .unwrap_err()
        .to_string()
        .contains("native MLP Q-head output_dim must be a multiple of 8"));
    }
}

#[test]
fn malformed_requests_are_rejected() {
    assert!(resolve(&request("auto", "mlp", Some("fp16"), 8, vec![]))
        .unwrap_err()
        .to_string()
        .contains("requires at least one GPU"));
    assert!(resolve(&request("python", "mlp", Some("fp16"), 8, vec![device(86, "A100")]))
        .unwrap_err()
        .to_string()
        .contains("inference_backend must be"));
}

#[test]
fn selected_backend_carries_its_existing_inference_profile() {
    let cutlass = resolve_plan(&request(
        "auto",
        "mlp",
        Some("fp16"),
        8,
        vec![device(86, "NVIDIA GeForce RTX 3060")],
    ))
    .unwrap();
    assert_eq!(cutlass.backend, Backend::Cutlass);
    assert_eq!(cutlass.policy, "cutlass-default-v1");
    assert_eq!(cutlass.inference_row_limit, 8192);

    let libtorch = resolve_plan(&request(
        "auto",
        "mlp",
        Some("fp16"),
        8,
        vec![device(75, "Tesla T4")],
    ))
    .unwrap();
    assert_eq!(libtorch.backend, Backend::LibTorch);
    assert_eq!(libtorch.policy, "libtorch-conservative-v1");
    assert_eq!(libtorch.inference_row_limit, 2048);
}
