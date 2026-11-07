//! Bounded fresh full-step comparisons; numeric widths never enter this policy.
use crate::{Error, Result};
use beam_core::calibration::{estimate, Estimate, Measurement};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub fn require_full_step(beam: u64, world: u32, pipeline: &Value) -> Result<Estimate> {
    if pipeline["phase"] != "pipeline_measured"
        || pipeline["pipeline_verified"] != true
        || pipeline["measurement_scope"] != "full_requested_frontier"
        || pipeline["requested_beam_effective"].as_u64() != Some(beam)
    {
        return Err(Error::Message(
            "working frontier requires verified full steps at the exact width".into(),
        ));
    }
    let profile = pipeline["selection"]
        .as_str()
        .ok_or_else(|| Error::Message("missing full-step selection".into()))?;
    let samples: Vec<Measurement> = serde_json::from_value(pipeline["measurements"].clone())?;
    if samples.iter().any(|r| {
        r.parents != beam || r.seconds_by_rank.len() != world as usize || r.profile != profile
    }) {
        return Err(Error::Message(
            "full-step cohort/width/profile mismatch".into(),
        ));
    }
    let result = estimate(profile, &samples)?;
    if json!(result) != pipeline["estimate"] {
        return Err(Error::Message(
            "full-step estimate differs from raw measurements".into(),
        ));
    }
    Ok(result)
}
#[derive(Debug, Serialize, Deserialize)]
pub struct WorkingSelection {
    pub beam: u64,
    pub pipeline: Value,
    pub efficiency_verified: bool,
    pub comparisons: Vec<Value>,
    pub scope: String,
}
pub fn choose(
    beam: u64,
    pipeline: Value,
    micro: u64,
    world: u32,
    mut measure: impl FnMut(u64, &Value) -> Result<Value>,
) -> Result<WorkingSelection> {
    let minimum = micro
        .checked_mul(world as u64)
        .filter(|n| *n > 0)
        .ok_or_else(|| Error::Message("invalid saturated working comparison".into()))?;
    let (mut current, mut pipeline, mut comparisons) = (beam, pipeline, vec![]);
    for _ in 0..3 {
        let upper = require_full_step(current, world, &pipeline)?;
        let width = current / 2;
        if width < minimum {
            return Err(Error::Message(
                "working boundary has no smaller saturated comparison".into(),
            ));
        }
        let reference = measure(width, &pipeline)?;
        if reference["environment"] != pipeline["environment"] {
            return Err(Error::Message(
                "working reference changed the frozen execution profile".into(),
            ));
        }
        let lower = require_full_step(width, world, &reference)?;
        let retention = lower.median / upper.median;
        let near_linear = retention >= 0.8;
        comparisons.push(json!({"tested_beam":current,"reference_beam":width,
            "seconds_per_parent":upper.median,"reference_seconds_per_parent":lower.median,
            "throughput_retention":retention,"near_linear":near_linear,
            "policy":"retain at least 80% of smaller measured-width throughput"}));
        if near_linear {
            return Ok(WorkingSelection {
                beam: current,
                pipeline,
                efficiency_verified: true,
                comparisons,
                scope: "largest suitable measured width; not an exhaustive efficient maximum"
                    .into(),
            });
        }
        current = width;
        pipeline = reference;
    }
    Err(Error::Message(
        "bounded working-frontier comparison did not find stable scaling".into(),
    ))
}
