//! Independent frozen-profile capacity searches; admission is not execution.
use crate::{
    calibration::{find_capacity, CapacityResult, Environment},
    Error, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Config {
    pub upper_bound: u64,
    pub alignment: u64,
    pub world: u32,
    pub outer: u64,
    pub minimum_flush: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Discovery {
    #[serde(flatten)]
    pub capacity: CapacityResult,
    pub memory_ceiling: CapacityResult,
    pub rejected: Vec<Value>,
    pub wall_seconds: f64,
    pub capacity_scope: String,
    pub flush_policy: String,
}
pub fn discover(
    config: &Config,
    working: bool,
    deadline: Instant,
    mut admit: impl FnMut(u64, &Environment) -> Result<Vec<Value>>,
) -> Result<Discovery> {
    if config.alignment == 0 || config.world == 0 || config.outer == 0 {
        return Err(Error::Message(
            "invalid maximum frontier search geometry".into(),
        ));
    }
    let started = Instant::now();
    let minimum = config
        .alignment
        .checked_mul(8)
        .ok_or_else(|| Error::Message("flush overflow".into()))?;
    let (mut best, mut memory_best): (Option<CapacityResult>, Option<CapacityResult>) =
        (None, None);
    let (mut probes, mut complete, mut rejected) = (0u64, true, vec![]);
    for shards in 1..=128u64 {
        if Instant::now() >= deadline {
            complete = false;
            break;
        }
        let stride = (config.world as u128) * shards as u128 * config.alignment as u128;
        if stride > config.upper_bound as u128 {
            continue;
        }
        let mut profile: Environment = [
            ("BEAM_B_MICRO".into(), config.outer.to_string()),
            ("BEAM_SHARD_COUNT".into(), shards.to_string()),
            ("BEAM_STREAM3_RING_SLOTS".into(), "2".into()),
            ("BEAM_STREAM4_ACTIVE_SORT_SLOTS".into(), "1".into()),
            ("BEAM_STREAM4_BATCH_CANDIDATES".into(), minimum.to_string()),
        ]
        .into();
        let mut search = |upper, profile: &Environment| {
            find_capacity(
                std::slice::from_ref(profile),
                upper,
                stride as u64,
                deadline,
                |beam, env| {
                    probes += 1;
                    admit(beam, env)
                },
            )
        };
        let mut result = match search(config.upper_bound, &profile) {
            Ok(result) => result,
            Err(Error::CapacityRejected(reason)) => {
                rejected.push(json!({"shards":shards,"phase":"memory","reason":reason}));
                continue;
            }
            Err(error) => return Err(error),
        };
        complete &= result.search_complete;
        if memory_best
            .as_ref()
            .is_none_or(|r| result.effective_beam > r.effective_beam)
        {
            memory_best = Some(result.clone());
        }
        if working {
            let capacity = result
                .plans
                .iter()
                .map(|p| p["SHARD_CAPACITY_CANDIDATES"].as_u64())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| Error::Message("missing native shard capacity".into()))?
                .into_iter()
                .max()
                .unwrap_or(0);
            if capacity == 0 {
                return Err(Error::Message("zero native shard capacity".into()));
            }
            let flush =
                (capacity as u128).div_ceil(config.alignment as u128) * config.alignment as u128;
            let flush = u64::try_from(flush)
                .map_err(|_| Error::Message("working flush overflow".into()))?
                .max(minimum)
                .max(config.minimum_flush);
            // An explicit larger arrival batch is preserved as a floor. This is
            // allocation policy, never a semantic survivor or shard truncation.
            profile.insert("BEAM_STREAM4_BATCH_CANDIDATES".into(), flush.to_string());
            result = match search(result.effective_beam, &profile) {
                Ok(result) => result,
                Err(Error::CapacityRejected(reason)) => {
                    rejected.push(json!({"shards":shards,"phase":"working","reason":reason}));
                    continue;
                }
                Err(error) => return Err(error),
            };
        }
        complete &= result.search_complete;
        if best
            .as_ref()
            .is_none_or(|r| result.effective_beam > r.effective_beam)
        {
            best = Some(result);
        }
        if !complete {
            break;
        }
    }
    let mut capacity = best.ok_or_else(|| {
        Error::CapacityRejected(
            "no model/GPU frontier was admitted within the capacity budget".into(),
        )
    })?;
    capacity.probes = probes;
    capacity.search_complete = complete;
    Ok(Discovery {
        capacity,
        memory_ceiling: memory_best.unwrap(),
        rejected,
        wall_seconds: started.elapsed().as_secs_f64(),
        capacity_scope: "native memory admission; full-step verification pending".into(),
        flush_policy: if working {
            "at least one admitted survivor slab and explicit arrival-batch floor"
        } else {
            "minimum allocation flush"
        }
        .into(),
    })
}
