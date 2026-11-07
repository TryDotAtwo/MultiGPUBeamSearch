//! Reproducible statistical selection. Component timings never promote a kernel.
use crate::ContractError;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
type Result<T> = std::result::Result<T, ContractError>;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurement {
    pub profile: String,
    pub parents: u64,
    pub seconds_by_rank: Vec<f64>,
    pub correctness_passed: bool,
    pub memory_admitted: bool,
    #[serde(default)]
    pub throttled: bool,
}
impl Measurement {
    pub fn seconds_per_parent(&self) -> Result<f64> {
        if self.parents == 0
            || self.seconds_by_rank.is_empty()
            || self
                .seconds_by_rank
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.)
        {
            return Err(ContractError::Invalid("parent count or rank timing"));
        }
        Ok(self.seconds_by_rank.iter().copied().fold(0., f64::max) / self.parents as f64)
    }
    fn validate(&self) -> Result<()> {
        self.seconds_per_parent()?;
        if !self.correctness_passed || !self.memory_admitted || self.throttled {
            return Err(ContractError::Invalid(
                "incorrect, unadmitted or throttled profile",
            ));
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Estimate {
    pub profile: String,
    pub median: f64,
    pub low: f64,
    pub high: f64,
    pub relative_mad: f64,
    pub repeats: usize,
}
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let half = values.len() / 2;
    if values.len() % 2 == 0 {
        values[half - 1] * 0.5 + values[half] * 0.5
    } else {
        values[half]
    }
}
// Versioned deterministic PRNG independent of platform and Python's RNG ABI.
fn random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
fn interval(values: &[f64]) -> (f64, f64) {
    let mut state = 0;
    let mut bootstrap = Vec::with_capacity(2000);
    let mut sample = vec![0.; values.len()];
    for _ in 0..2000 {
        for value in &mut sample {
            let index = ((random(&mut state) as u128 * values.len() as u128) >> 64) as usize;
            *value = values[index];
        }
        bootstrap.push(median(&mut sample));
    }
    bootstrap.sort_by(f64::total_cmp);
    (bootstrap[50], bootstrap[1950])
}
fn summarize(profile: &str, mut values: Vec<f64>) -> Result<Estimate> {
    if values.len() < 5 {
        return Err(ContractError::Invalid("insufficient measured repeats"));
    }
    let center = median(&mut values);
    let mut deviations: Vec<_> = values.iter().map(|v| (v - center).abs()).collect();
    let relative_mad = median(&mut deviations) / center;
    if !relative_mad.is_finite() || relative_mad > 0.10 {
        return Err(ContractError::Invalid("unstable timing samples"));
    }
    let (low, high) = interval(&values);
    Ok(Estimate {
        profile: profile.into(),
        median: center,
        low,
        high,
        relative_mad,
        repeats: values.len(),
    })
}
pub fn estimate(profile: &str, samples: &[Measurement]) -> Result<Estimate> {
    let selected: Vec<_> = samples.iter().filter(|s| s.profile == profile).collect();
    let first = selected
        .first()
        .ok_or(ContractError::Invalid("no profile measurements"))?;
    let mut values = Vec::new();
    for row in &selected {
        row.validate()?;
        if (row.parents, row.seconds_by_rank.len()) != (first.parents, first.seconds_by_rank.len())
        {
            return Err(ContractError::Invalid("mixed workload or rank cohort"));
        }
        values.push(row.seconds_per_parent()?);
    }
    summarize(profile, values)
}
pub fn select(
    samples: &[Measurement],
    baseline: &str,
    min_improvement: f64,
) -> Result<(Estimate, BTreeMap<String, String>)> {
    if !min_improvement.is_finite() || !(0.0..1.0).contains(&min_improvement) {
        return Err(ContractError::Invalid("improvement threshold"));
    }
    let cohorts: BTreeSet<_> = samples
        .iter()
        .map(|s| (s.parents, s.seconds_by_rank.len()))
        .collect();
    if cohorts.len() != 1 {
        return Err(ContractError::Invalid(
            "all candidates must share workload and rank cohort",
        ));
    }
    let mut chosen = estimate(baseline, samples)?;
    let mut rejected = BTreeMap::new();
    let mut valid = Vec::new();
    for name in samples
        .iter()
        .map(|s| s.profile.as_str())
        .collect::<BTreeSet<_>>()
    {
        if name == baseline {
            continue;
        }
        match estimate(name, samples) {
            Ok(value) => valid.push(value),
            Err(error) => {
                rejected.insert(name.into(), error.to_string());
            }
        }
    }
    valid.sort_by(|a, b| {
        a.median
            .total_cmp(&b.median)
            .then_with(|| a.profile.cmp(&b.profile))
    });
    for candidate in valid {
        if candidate.high < chosen.low * (1. - min_improvement) {
            chosen = candidate;
            break;
        }
        rejected.insert(
            candidate.profile,
            "no statistically separated material gain".into(),
        );
    }
    Ok((chosen, rejected))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Workload {
    pub model: String,
    pub input: String,
    pub beam: u64,
    pub precision: String,
    pub ranks: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PairedMeasurement {
    pub baseline_workload: Workload,
    pub candidate_workload: Workload,
    pub baseline: Measurement,
    pub candidate: Measurement,
    pub baseline_peak_bytes: u64,
    pub candidate_peak_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Promotion {
    pub policy: String,
    pub accepted: bool,
    pub reason: String,
    pub pairs: usize,
    pub time_ratio: Option<Estimate>,
}
/// Inputs are paired warmed *complete-step* measurements; callers keep raw
/// provenance, cold/session timings, telemetry and component timings separately.
pub fn paired_promotion(rows: &[PairedMeasurement]) -> Result<Promotion> {
    let mut decision = Promotion {
        policy: "paired-full-step-v1-splitmix64".into(),
        accepted: false,
        reason: "at least five warmed paired repeats required".into(),
        pairs: rows.len(),
        time_ratio: None,
    };
    if rows.len() < 5 {
        return Ok(decision);
    }
    let workload = &rows[0].baseline_workload;
    if workload.model.is_empty()
        || workload.input.is_empty()
        || workload.precision.is_empty()
        || workload.beam == 0
        || workload.ranks == 0
    {
        return Err(ContractError::Invalid("benchmark workload identity"));
    }
    let mut ratios = Vec::new();
    let mut memory_reduced = true;
    for row in rows {
        if &row.baseline_workload != workload || &row.candidate_workload != workload {
            return Err(ContractError::Invalid("paired workload identity mismatch"));
        }
        for sample in [&row.baseline, &row.candidate] {
            sample.validate()?;
            if sample.parents != workload.beam
                || sample.seconds_by_rank.len() != workload.ranks as usize
            {
                return Err(ContractError::Invalid(
                    "paired complete-step workload/rank count",
                ));
            }
        }
        if row.baseline_peak_bytes == 0 || row.candidate_peak_bytes == 0 {
            return Err(ContractError::Invalid("missing peak memory measurement"));
        }
        memory_reduced &= row.candidate_peak_bytes < row.baseline_peak_bytes;
        let ratio = row.candidate.seconds_per_parent()? / row.baseline.seconds_per_parent()?;
        if !ratio.is_finite() || ratio <= 0. {
            return Err(ContractError::Invalid("paired timing ratio"));
        }
        ratios.push(ratio);
    }
    match summarize("candidate/baseline", ratios) {
        Ok(estimate) => {
            if estimate.high <= 0.95 {
                decision.accepted = true;
                decision.reason = "full step at least 5% faster".into();
            } else if memory_reduced && estimate.high <= 1.03 {
                decision.accepted = true;
                decision.reason = "peak memory reduced; full-step slowdown at most 3%".into();
            } else {
                decision.reason = "no stable full-step gain or qualifying memory reduction".into();
            }
            decision.time_ratio = Some(estimate);
        }
        Err(error) => decision.reason = error.to_string(),
    }
    Ok(decision)
}
