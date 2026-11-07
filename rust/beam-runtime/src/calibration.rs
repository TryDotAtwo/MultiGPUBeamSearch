//! Bounded control loops for exact native admission and complete-step selection.
use crate::{Error, Result};
use beam_core::{
    calibration::{select, Estimate, Measurement},
    planning::{validate_rank_plans, RankPlan, TUNABLE_KEYS},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};
pub type Environment = BTreeMap<String, String>;

fn validate(plans: &[Value], beam: u64) -> Result<()> {
    let typed: Vec<RankPlan> = plans
        .iter()
        .cloned()
        .map(serde_json::from_value)
        .collect::<std::result::Result<_, _>>()?;
    validate_rank_plans(&typed, beam)?;
    Ok(())
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapacityResult {
    pub requested_beam: u64,
    pub effective_beam: u64,
    pub profile: Environment,
    pub plans: Vec<Value>,
    pub probes: u64,
    pub upper_rejected_beam: Option<u64>,
    pub search_complete: bool,
    pub full_step_verified: bool,
}
/// Native admission must be monotone for each supplied frozen profile and
/// memory snapshot. Only an explicit capacity rejection narrows the upper bound.
pub fn find_capacity(
    profiles: &[Environment],
    upper_bound: u64,
    alignment: u64,
    deadline: Instant,
    mut admit: impl FnMut(u64, &Environment) -> Result<Vec<Value>>,
) -> Result<CapacityResult> {
    if alignment == 0 || upper_bound < alignment || profiles.is_empty() {
        return Err(Error::Message(
            "invalid capacity search ceiling/alignment/profiles".into(),
        ));
    }
    let maximum = (upper_bound / alignment) as u128;
    let mut best: Option<CapacityResult> = None;
    let mut probes = 0u64;
    let mut complete = true;
    for profile in profiles {
        let (mut low, mut high) = (0u128, maximum + 1);
        let mut accepted = None;
        let mut rejected = None;
        while high - low > 1 {
            if Instant::now() >= deadline {
                complete = false;
                break;
            }
            let units = (low + high) / 2;
            let beam = (units * alignment as u128) as u64;
            probes = probes
                .checked_add(1)
                .ok_or_else(|| Error::Message("capacity probe counter overflow".into()))?;
            match admit(beam, profile) {
                Ok(plans) => {
                    validate(&plans, beam)?;
                    low = units;
                    accepted = Some(plans);
                }
                Err(Error::CapacityRejected(_)) => {
                    high = units;
                    rejected = Some(beam);
                }
                Err(error) => return Err(error),
            }
        }
        let beam = (low * alignment as u128) as u64;
        if let Some(plans) = accepted {
            if best.as_ref().is_none_or(|r| beam > r.requested_beam) {
                best = Some(CapacityResult {
                    requested_beam: beam,
                    effective_beam: beam,
                    profile: profile.clone(),
                    plans,
                    probes,
                    upper_rejected_beam: rejected,
                    search_complete: complete,
                    full_step_verified: false,
                });
            }
        }
        if !complete {
            break;
        }
    }
    let mut best = best.ok_or_else(|| {
        Error::CapacityRejected("no frontier admitted within search budget".into())
    })?;
    best.probes = probes;
    best.search_complete = complete;
    Ok(best)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    pub environment: Environment,
}
pub trait ProfileEvaluator {
    fn admit(&mut self, environment: &Environment) -> Result<Vec<Value>>;
    fn finish_planning(&mut self) -> Result<()>;
    fn measure(&mut self, environment: &Environment, plans: &[Value]) -> Result<Vec<Measurement>>;
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Selection {
    pub environment: Environment,
    pub estimate: Estimate,
    pub rejected: BTreeMap<String, String>,
    pub effective_beam: u64,
    pub workload_parents: u64,
    pub world_size: usize,
    pub measurements: Vec<Measurement>,
    pub planning_scope: String,
    pub search_policy: String,
}
/// Runs all admissions, releases planning allocations, then owns the complete
/// measurement/selection loop. No callback receives a GPU or per-candidate pointer.
pub fn tune_profiles(
    evaluator: &mut impl ProfileEvaluator,
    profiles: &[Profile],
    beam: u64,
    micro: u64,
    deadline: Instant,
) -> Result<Selection> {
    if beam == 0 || micro == 0 || profiles.is_empty() || profiles[0].name != "baseline" {
        return Err(Error::Message(
            "baseline/exact beam/frozen inference microbatch required".into(),
        ));
    }
    let mut names = BTreeSet::new();
    for profile in profiles {
        if profile.name.is_empty() || !names.insert(&profile.name) {
            return Err(Error::Message("duplicate or empty profile name".into()));
        }
        if profile.environment.get("BEAM_ENSEMBLE_INFERENCE_MICRO") != Some(&micro.to_string()) {
            return Err(Error::Message(
                "downstream cannot change frozen inference microbatch".into(),
            ));
        }
        for key in profile.environment.keys() {
            if key != "BEAM_ENSEMBLE_INFERENCE_MICRO" && !TUNABLE_KEYS.contains(&key.as_str()) {
                return Err(Error::Message(format!(
                    "profile cannot change semantic/capacity policy {key}"
                )));
            }
        }
    }
    let mut admitted = Vec::new();
    let mut rejected = BTreeMap::new();
    let admission = (|| {
        for profile in profiles {
            if Instant::now() >= deadline {
                break;
            }
            let result = evaluator.admit(&profile.environment).and_then(|p| {
                validate(&p, beam)?;
                Ok(p)
            });
            match result {
                Ok(plans) => admitted.push((profile, plans)),
                Err(error @ (Error::CapacityRejected(_) | Error::Contract(_)))
                    if profile.name != "baseline" =>
                {
                    rejected.insert(profile.name.clone(), error.to_string());
                }
                Err(error) => return Err(error),
            }
        }
        if admitted.first().is_none_or(|(p, _)| p.name != "baseline") {
            return Err(Error::Message(
                "no admitted baseline within calibration deadline".into(),
            ));
        }
        Ok(())
    })();
    let cleanup = evaluator.finish_planning();
    admission?;
    cleanup?;
    let world = admitted[0].1.len();
    let mut measurements = Vec::new();
    for (profile, plans) in &admitted {
        if profile.name != "baseline" && Instant::now() >= deadline {
            break;
        }
        let result = evaluator
            .measure(&profile.environment, plans)
            .and_then(|mut rows| {
                if rows.is_empty()
                    || rows
                        .iter()
                        .any(|r| r.parents != beam || r.seconds_by_rank.len() != world)
                {
                    return Err(beam_core::ContractError::Invalid(
                        "full-step measurement does not match admitted frontier/cohort",
                    )
                    .into());
                }
                for row in &mut rows {
                    row.profile = profile.name.clone();
                }
                Ok(rows)
            });
        match result {
            Ok(rows) => measurements.extend(rows),
            Err(error @ (Error::CapacityRejected(_) | Error::Contract(_)))
                if profile.name != "baseline" =>
            {
                rejected.insert(profile.name.clone(), error.to_string());
            }
            Err(error) => return Err(error),
        }
    }
    let (estimate, statistics) = select(&measurements, "baseline", 0.05)?;
    rejected.extend(statistics);
    let environment = profiles
        .iter()
        .find(|p| p.name == estimate.profile)
        .ok_or_else(|| Error::Message("selected unknown profile".into()))?
        .environment
        .clone();
    Ok(Selection {
        environment,
        estimate,
        rejected,
        effective_beam: beam,
        workload_parents: beam,
        world_size: world,
        measurements,
        planning_scope: "exact_requested_frontier".into(),
        search_policy: "rust-full-step-shortlist-v1".into(),
    })
}
