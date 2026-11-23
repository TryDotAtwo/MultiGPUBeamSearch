//! Versioned permutation runtime phase selection shared by Python and Rust launchers.
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;
pub const EXECUTION_PLAN_SCHEMA_VERSION: u32 = 1;
pub const MAXIMUM_CALIBRATION_BEAM_WIDTH: u64 = 1_048_576;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermutationSearchPhaseRequest {
    pub autotune: bool,
    pub model_backend: String,
    pub calibration_binary_name: Option<String>,
    pub maximum_requested: bool,
    pub discover_capacity: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum RequestedBeamWidth {
    Exact(u64),
    Token(String),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermutationSearchExecutionPlanRequest {
    pub autotune: bool,
    pub model_backend: String,
    pub calibration_binary_name: Option<String>,
    pub requested_beam_width: RequestedBeamWidth,
    pub discover_capacity: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PermutationSearchPhase {
    InferenceCalibration,
    CapacityDiscovery,
    MaximumFrontierSearch,
    PipelineCalibration,
    WorkingCapacitySelection,
    Execute,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PermutationSearchPhasePlan {
    pub schema_version: u32,
    pub phases: Vec<PermutationSearchPhase>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PermutationSearchExecutionPlan {
    pub schema_version: u32,
    pub phases: Vec<PermutationSearchPhase>,
    pub calibration_beam_width: u64,
}

/// Resolve the existing calibration and capacity branches without changing their implementations.
pub fn resolve(request: PermutationSearchPhaseRequest) -> PermutationSearchPhasePlan {
    let inference_calibration = request.autotune
        && (request.model_backend == "ensemble"
            || (request.model_backend == "mlp"
                && matches!(
                    request.calibration_binary_name.as_deref(),
                    Some("stream1_native_mlp_benchmark" | "stream1_libtorch_mlp_benchmark")
                )));

    let mut phases = Vec::with_capacity(6);
    if inference_calibration {
        phases.push(PermutationSearchPhase::InferenceCalibration);
    }
    if request.discover_capacity && !request.maximum_requested {
        phases.push(PermutationSearchPhase::CapacityDiscovery);
    }
    if request.maximum_requested {
        phases.push(PermutationSearchPhase::MaximumFrontierSearch);
    }
    if inference_calibration {
        phases.push(PermutationSearchPhase::PipelineCalibration);
        if request.maximum_requested {
            phases.push(PermutationSearchPhase::WorkingCapacitySelection);
        }
    }
    phases.push(PermutationSearchPhase::Execute);

    PermutationSearchPhasePlan {
        schema_version: SCHEMA_VERSION,
        phases,
    }
}

/// Resolve the public requested width into the unchanged width used by preparation.
pub fn resolve_execution_plan(
    request: PermutationSearchExecutionPlanRequest,
) -> std::result::Result<PermutationSearchExecutionPlan, String> {
    let (maximum_requested, calibration_beam_width) = match request.requested_beam_width {
        RequestedBeamWidth::Exact(width) if width > 0 => (false, width),
        RequestedBeamWidth::Exact(_) => return Err("beam_width must be positive".into()),
        RequestedBeamWidth::Token(token) if token == "max" => {
            if !request.autotune {
                return Err("maximum beam requires autotuning".into());
            }
            (true, MAXIMUM_CALIBRATION_BEAM_WIDTH)
        }
        RequestedBeamWidth::Token(_) => {
            return Err("beam_width must be a positive integer or 'max'".into());
        }
    };
    let phases = resolve(PermutationSearchPhaseRequest {
        autotune: request.autotune,
        model_backend: request.model_backend,
        calibration_binary_name: request.calibration_binary_name,
        maximum_requested,
        discover_capacity: request.discover_capacity,
    })
    .phases;
    Ok(PermutationSearchExecutionPlan {
        schema_version: EXECUTION_PLAN_SCHEMA_VERSION,
        phases,
        calibration_beam_width,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(
        autotune: bool,
        model_backend: &str,
        calibration_binary_name: Option<&str>,
        maximum_requested: bool,
        discover_capacity: bool,
    ) -> Vec<PermutationSearchPhase> {
        resolve(PermutationSearchPhaseRequest {
            autotune,
            model_backend: model_backend.into(),
            calibration_binary_name: calibration_binary_name.map(str::to_owned),
            maximum_requested,
            discover_capacity,
        })
        .phases
    }

    #[test]
    fn preserves_inference_and_pipeline_calibration_predicates() {
        use PermutationSearchPhase::*;

        assert_eq!(
            plan(
                true,
                "mlp",
                Some("stream1_native_mlp_benchmark"),
                false,
                false
            ),
            vec![InferenceCalibration, PipelineCalibration, Execute]
        );
        assert_eq!(
            plan(
                true,
                "mlp",
                Some("stream1_cutlass_mlp_benchmark"),
                false,
                false
            ),
            vec![Execute]
        );
        assert_eq!(
            plan(true, "ensemble", None, false, false),
            vec![InferenceCalibration, PipelineCalibration, Execute]
        );
        assert_eq!(plan(true, "transformer", None, false, false), vec![Execute]);
        assert_eq!(
            plan(
                false,
                "mlp",
                Some("stream1_native_mlp_benchmark"),
                false,
                false
            ),
            vec![Execute]
        );
    }

    #[test]
    fn preserves_capacity_and_maximum_stage_order_and_failure_boundary() {
        use PermutationSearchPhase::*;

        assert_eq!(
            plan(
                true,
                "mlp",
                Some("stream1_libtorch_mlp_benchmark"),
                false,
                true
            ),
            vec![
                InferenceCalibration,
                CapacityDiscovery,
                PipelineCalibration,
                Execute
            ]
        );
        assert_eq!(
            plan(
                true,
                "mlp",
                Some("stream1_libtorch_mlp_benchmark"),
                true,
                true
            ),
            vec![
                InferenceCalibration,
                MaximumFrontierSearch,
                PipelineCalibration,
                WorkingCapacitySelection,
                Execute
            ]
        );
        // Preserve the existing delayed unsupported-calibrator failure at the
        // maximum-frontier phase, after preparation and runner snapshotting.
        assert_eq!(
            plan(true, "transformer", None, true, false),
            vec![MaximumFrontierSearch, Execute]
        );
        // Fixed-width capacity discovery likewise remains a requested phase;
        // the caller raises at its former boundary when calibration is absent.
        assert_eq!(
            plan(true, "transformer", None, false, true),
            vec![CapacityDiscovery, Execute]
        );
    }
    
    #[test]
    fn execution_plan_preserves_exact_width_and_maximum_calibration_width() {
        let request = |requested_beam_width, autotune| PermutationSearchExecutionPlanRequest {
            autotune,
            model_backend: "transformer".into(),
            calibration_binary_name: None,
            requested_beam_width,
            discover_capacity: false,
        };
        let exact =
            resolve_execution_plan(request(RequestedBeamWidth::Exact(37), false)).unwrap();
        assert_eq!(exact.calibration_beam_width, 37);
        assert_eq!(exact.phases, vec![PermutationSearchPhase::Execute]);

        let maximum = resolve_execution_plan(request(
            RequestedBeamWidth::Token("max".into()),
            true,
        ))
        .unwrap();
        assert_eq!(
            maximum.calibration_beam_width,
            MAXIMUM_CALIBRATION_BEAM_WIDTH
        );
        assert_eq!(
            maximum.phases,
            vec![
                PermutationSearchPhase::MaximumFrontierSearch,
                PermutationSearchPhase::Execute
            ]
        );
        assert!(resolve_execution_plan(request(
            RequestedBeamWidth::Token("maximum".into()),
            true
        ))
        .is_err());
        assert!(resolve_execution_plan(request(
            RequestedBeamWidth::Token("max".into()),
            false
        ))
        .is_err());
    }

}
