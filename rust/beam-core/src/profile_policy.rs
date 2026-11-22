//! Bounded scheduling proposals. Only the native planner can admit memory.
use crate::{components::ServiceEnvelope, planning::TUNABLE_KEYS, ContractError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
pub type Environment = BTreeMap<String, String>;
pub const POLICY: &str = "rust-native-admitted-shortlist-v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hardware {
    pub sm_count: u32,
    pub l2_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchGeometry {
    pub beam: u64,
    pub world: u32,
    pub moves: u32,
    pub alignment: u64,
    pub row_factor: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposedProfile {
    pub name: String,
    pub environment: Environment,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proposals {
    pub candidates: Vec<ProposedProfile>,
    pub omitted: Vec<String>,
    pub regime: String,
    pub policy: String,
}
fn invalid() -> ContractError {
    ContractError::Invalid("downstream scheduling proposal")
}
pub fn validate_environment(env: &Environment, micro: u64) -> Result<(), ContractError> {
    if env.len() != 6
        || !(1..=65536).contains(&micro)
        || env.get("BEAM_ENSEMBLE_INFERENCE_MICRO") != Some(&micro.to_string())
    {
        return Err(invalid());
    }
    for (key, value) in env {
        if (!TUNABLE_KEYS.contains(&key.as_str()) && key != "BEAM_ENSEMBLE_INFERENCE_MICRO")
            || value.is_empty()
            || !value.bytes().all(|c| c.is_ascii_digit())
            || !value.parse::<u64>().is_ok_and(|n| n > 0 && n <= 1 << 40)
        {
            return Err(invalid());
        }
    }
    Ok(())
}
fn round_up(n: u128, alignment: u64) -> Result<u64, ContractError> {
    u64::try_from(n.div_ceil(alignment as u128) * alignment as u128)
        .map_err(|_| ContractError::Overflow("profile rounding"))
}
pub fn proposals(
    shape: &SearchGeometry,
    baseline: &Environment,
    hardware: &Hardware,
    service: &ServiceEnvelope,
    fine: bool,
    seen: &BTreeSet<Environment>,
) -> Result<Proposals, ContractError> {
    if shape.beam == 0
        || shape.world == 0
        || shape.world > 65535
        || !(1..=255).contains(&shape.moves)
        || shape.alignment == 0
        || (shape.row_factor != 1 && shape.row_factor != shape.moves as u64)
        || hardware.sm_count == 0
    {
        return Err(invalid());
    }
    let value = |key: &str| {
        baseline
            .get(key)
            .and_then(|s| s.parse::<u64>().ok())
            .ok_or_else(invalid)
    };
    let micro = value("BEAM_ENSEMBLE_INFERENCE_MICRO")?;
    validate_environment(baseline, micro)?;
    let (outer, shards, staging, lanes, batch) = (
        value("BEAM_B_MICRO")?,
        value("BEAM_SHARD_COUNT")?,
        value("BEAM_STREAM3_RING_SLOTS")?,
        value("BEAM_STREAM4_ACTIVE_SORT_SLOTS")?,
        value("BEAM_STREAM4_BATCH_CANDIDATES")?,
    );
    let quantum = micro
        .checked_mul(shape.row_factor)
        .ok_or(ContractError::Overflow("inference row quantum"))?;
    if outer % quantum != 0 {
        return Err(invalid());
    }
    let local = shape.beam.div_ceil(shape.world as u64);
    let max_shards = local.div_ceil(shape.alignment).clamp(1, 128);
    let max_lanes = (hardware.sm_count as u64).div_ceil(16).clamp(1, max_shards);
    let mut pool = Vec::new();
    let mut add = |name: String, updates: Vec<(&str, u64)>| {
        let mut environment = baseline.clone();
        for (key, value) in updates {
            environment.insert(key.into(), value.to_string());
        }
        pool.push(ProposedProfile { name, environment });
    };
    if fine {
        for s in [
            shards.saturating_sub(1).max(1),
            (shards + 1).min(max_shards),
        ] {
            add(
                format!("shards-{s}"),
                vec![
                    ("BEAM_SHARD_COUNT", s),
                    ("BEAM_STREAM4_ACTIVE_SORT_SLOTS", lanes.min(s)),
                ],
            );
        }
        for n in [lanes.saturating_sub(1).max(1), (lanes + 1).min(max_lanes)] {
            add(
                format!("lanes-{n}"),
                vec![("BEAM_STREAM4_ACTIVE_SORT_SLOTS", n)],
            );
        }
        for n in [staging.saturating_sub(1).max(2), (staging + 1).min(8)] {
            add(format!("staging-{n}"), vec![("BEAM_STREAM3_RING_SLOTS", n)]);
        }
        let flush = round_up(batch as u128 * 5 / 4, shape.alignment)?;
        add(
            format!("batch-{flush}"),
            vec![("BEAM_STREAM4_BATCH_CANDIDATES", flush)],
        );
        for numerator in [3, 5] {
            let n = round_up(outer as u128 * numerator / 4, quantum)?.max(quantum);
            if n as u128 <= (local as u128 * shape.row_factor as u128).max(quantum as u128) {
                add(format!("outer-{n}"), vec![("BEAM_B_MICRO", n)]);
            }
        }
    } else {
        for factor in [2, 4] {
            let n = outer as u128 * factor;
            if n <= local as u128 * shape.row_factor as u128 && n <= u64::MAX as u128 {
                add(
                    format!("outer-{n}"),
                    vec![
                        ("BEAM_B_MICRO", n as u64),
                        (
                            "BEAM_STREAM3_RING_SLOTS",
                            staging.div_ceil(factor as u64).max(2),
                        ),
                    ],
                );
            }
        }
        for n in [2, 4, 8] {
            add(format!("staging-{n}"), vec![("BEAM_STREAM3_RING_SLOTS", n)]);
        }
        for n in [1, max_lanes, (lanes * 2).min(max_lanes)] {
            add(
                format!("lanes-{n}"),
                vec![("BEAM_STREAM4_ACTIVE_SORT_SLOTS", n.min(shards))],
            );
        }
        let bulk = round_up(
            (local as u128 * shape.moves as u128).div_ceil(shards as u128 * 4),
            shape.alignment,
        )?;
        add("bulk".into(), vec![("BEAM_STREAM4_BATCH_CANDIDATES", bulk)]);
        if hardware.l2_bytes > 0 {
            let n = round_up((hardware.l2_bytes / 32).max(1) as u128, shape.alignment)?;
            add(
                "l2-batch".into(),
                vec![("BEAM_STREAM4_BATCH_CANDIDATES", n)],
            );
        }
        for s in [
            1,
            (shards * 2).min(max_shards),
            (shards / 2).max(1),
            (hardware.sm_count as u64).div_ceil(8).min(max_shards),
        ] {
            add(
                format!("shards-{s}"),
                vec![
                    ("BEAM_SHARD_COUNT", s),
                    ("BEAM_STREAM4_ACTIVE_SORT_SLOTS", lanes.min(s)),
                ],
            );
        }
    }
    let ratio = service.downstream_service_work_seconds / service.inference_seconds;
    if !ratio.is_finite() || ratio < 0. || service.inference_seconds <= 0. {
        return Err(invalid());
    }
    let (regime, priority) = if ratio <= 0.05 {
        (
            "inference_dominated",
            [
                "outer-", "staging-", "lanes-", "bulk", "shards-", "l2-batch",
            ],
        )
    } else if service.stream4_seconds >= service.downstream_service_work_seconds / 2. {
        (
            "sort_pressure",
            [
                "bulk", "shards-", "lanes-", "outer-", "staging-", "l2-batch",
            ],
        )
    } else {
        (
            "transport_or_mixed",
            [
                "outer-", "staging-", "bulk", "shards-", "lanes-", "l2-batch",
            ],
        )
    };
    let family = |name: &str| {
        priority
            .iter()
            .position(|p| name.starts_with(p))
            .unwrap_or(priority.len())
    };
    pool.sort_by_key(|r| family(&r.name));
    let mut used = seen.clone();
    used.insert(baseline.clone());
    let mut families = BTreeSet::new();
    let mut candidates = Vec::new();
    let mut omitted = Vec::new();
    for row in pool {
        validate_environment(&row.environment, micro)?;
        let flush = row.environment["BEAM_STREAM4_BATCH_CANDIDATES"]
            .parse::<u64>()
            .map_err(|_| invalid())?;
        if flush < batch || !used.insert(row.environment.clone()) {
            continue;
        }
        if candidates.len() >= if fine { 10 } else { 4 }
            || (!fine && !families.insert(family(&row.name)))
        {
            omitted.push(row.name);
            continue;
        }
        candidates.push(row);
    }
    Ok(Proposals {
        candidates,
        omitted,
        regime: regime.into(),
        policy: POLICY.into(),
    })
}
