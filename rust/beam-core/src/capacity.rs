//! Width policy and finite upper bounds. A multiset bound is not an orbit proof.
use crate::{search::PermutationGraph, ContractError};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeamWidth {
    Exact(u64),
    Maximum,
}
impl<'de> Deserialize<'de> for BeamWidth {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(d)?;
        if value == "max" {
            return Ok(Self::Maximum);
        }
        value
            .as_u64()
            .filter(|n| *n > 0)
            .map(Self::Exact)
            .ok_or_else(|| serde::de::Error::custom("beam width must be a positive integer or max"))
    }
}
impl Serialize for BeamWidth {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Exact(n) => s.serialize_u64(*n),
            Self::Maximum => s.serialize_str("max"),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrontierCeiling {
    pub upper_bound: u64,
    pub memory_state_ceiling: u64,
    pub explicit_state_ceiling: Option<u64>,
    pub excluded_states: u64,
    pub multiset_bound_clamped: bool,
    pub reachable_orbit_verified: bool,
}
pub fn frontier_ceiling(
    graph: &PermutationGraph,
    world: u32,
    storage: u64,
    gpu_bytes: &[u64],
    explicit: Option<u64>,
) -> Result<FrontierCeiling, ContractError> {
    graph.validate()?;
    if world == 0
        || gpu_bytes.len() != world as usize
        || gpu_bytes.contains(&0)
        || storage < graph.center.len() as u64
        || storage % 16 != 0
        || explicit == Some(0)
    {
        return Err(ContractError::Invalid("capacity cohort/storage/ceiling"));
    }
    let memory = ((*gpu_bytes.iter().min().unwrap() as u128 * world as u128) / storage as u128)
        .min(u64::MAX as u128) as u64;
    let cap = memory.min(explicit.unwrap_or(u64::MAX));
    let mut excluded = BTreeSet::from([graph.center.clone()]);
    for generator in &graph.generators {
        let mut inverse = vec![0; generator.len()];
        for (index, source) in generator.iter().enumerate() {
            inverse[*source as usize] = index;
        }
        excluded.insert(inverse.iter().map(|i| graph.center[*i]).collect());
    }
    // Counts never exceed cap + exclusions + 1. Each intermediate multiplication
    // is bounded by this u64-sized limit times the (at most 150) state length.
    let limit = cap as u128 + excluded.len() as u128 + 1;
    let mut counts = [0usize; 256];
    for value in &graph.center {
        counts[*value as usize] += 1;
    }
    let (mut arrangements, mut previous, mut clamped) = (1u128, 0usize, false);
    'counts: for count in counts.into_iter().filter(|n| *n > 0) {
        let mut choose = 1u128;
        for i in 1..=count {
            choose = choose * (previous + i) as u128 / i as u128;
            if choose * arrangements >= limit {
                arrangements = limit;
                clamped = true;
                break 'counts;
            }
        }
        arrangements *= choose;
        previous += count;
    }
    let available = arrangements.saturating_sub(excluded.len() as u128);
    Ok(FrontierCeiling {
        upper_bound: available.min(cap as u128) as u64,
        memory_state_ceiling: memory,
        explicit_state_ceiling: explicit,
        excluded_states: excluded.len() as u64,
        multiset_bound_clamped: clamped,
        reachable_orbit_verified: false,
    })
}
