//! Typed selection for the permutation runner implementation.
//!
//! Python/Torch observes the selected devices and passes their architecture/name
//! snapshot. Rust owns the route table and capability checks; search, scoring,
//! and kernels stay unchanged.
use crate::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceObservation {
    pub sm: u32,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub requested: String,
    pub model_backend: String,
    pub dtype: Option<String>,
    pub output_dim: u64,
    pub devices: Vec<DeviceObservation>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    LibTorch,
    Cutlass,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LibTorch => "libtorch",
            Self::Cutlass => "cutlass",
        }
    }
}

fn q_head_alignment_error(request: &Request) -> bool {
    request.model_backend == "mlp"
        && request.output_dim != 1
        && request.output_dim % 8 != 0
}

pub fn resolve(request: &Request) -> Result<Backend> {
    if !matches!(request.requested.as_str(), "auto" | "libtorch" | "cutlass") {
        return Err(Error::Message(
            "inference_backend must be auto, libtorch, or cutlass".into(),
        ));
    }
    if request.requested == "cutlass" && q_head_alignment_error(request) {
        return Err(Error::Message(
            "native MLP Q-head output_dim must be a multiple of 8 for CUTLASS row alignment; scalar output_dim=1 is supported".into(),
        ));
    }
    if request.devices.is_empty() {
        return Err(Error::Message(
            "inference backend selection requires at least one GPU".into(),
        ));
    }
    if request.dtype.as_deref() == Some("bf16")
        && request.devices.iter().any(|device| device.sm < 80)
    {
        return Err(Error::Message(
            "native BF16 Stream1 requires SM80+ on every selected GPU; use FP16 on SM75".into(),
        ));
    }
    if request.dtype.as_deref() == Some("fp16")
        && request.devices.iter().any(|device| device.sm < 75)
    {
        return Err(Error::Message(
            "this native FP16 MLP backend uses SM75+ TensorOp kernels; older GPU executors need a separate adapter".into(),
        ));
    }

    let selected = match request.requested.as_str() {
        "libtorch" => Backend::LibTorch,
        "cutlass" => Backend::Cutlass,
        "auto" => {
            let older_ensemble = request.model_backend == "ensemble"
                && request.devices.iter().any(|device| device.sm < 80);
            let has_t4 = request.devices.iter().any(|device| {
                device.name.split_whitespace().any(|token| token == "T4")
            });
            if older_ensemble || has_t4 {
                Backend::LibTorch
            } else {
                Backend::Cutlass
            }
        }
        _ => unreachable!("the requested backend was validated above"),
    };

    if request.model_backend == "ensemble"
        && selected == Backend::Cutlass
        && request.devices.iter().any(|device| device.sm < 80)
    {
        return Err(Error::Message(
            "ensemble CUTLASS requires SM80+ on every selected GPU".into(),
        ));
    }
    if selected == Backend::Cutlass && q_head_alignment_error(request) {
        return Err(Error::Message(
            "native MLP Q-head output_dim must be a multiple of 8 for CUTLASS row alignment; scalar output_dim=1 is supported".into(),
        ));
    }
    Ok(selected)
}
