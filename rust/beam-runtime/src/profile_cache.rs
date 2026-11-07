//! Calibration caches are evidence, never authority to skip native admission.
use crate::{artifacts, inference::validate_receipt, telemetry::Receipt, Error, Result};
use beam_core::calibration::{estimate, Measurement};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::Read,
    path::Path,
};
pub type FreeMemory = BTreeMap<String, u64>;

pub fn cache_key(identity: &Value) -> Result<String> {
    // serde_json uses ordered maps (preserve_order is not enabled).
    Ok(artifacts::sha256_bytes(&serde_json::to_vec(identity)?))
}
fn invalid() -> Error {
    Error::Message("invalid or incomplete cached inference evidence".into())
}
fn normalize(data: &Value, signature: &Value, max_batch: u64) -> Result<Value> {
    if data["signature"] != *signature || data["phase"] != "inference_verified" {
        return Err(invalid());
    }
    let batch = data["parent_batch"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= max_batch)
        .ok_or_else(invalid)?;
    let world = signature["world_size"]
        .as_u64()
        .filter(|n| *n > 0 && *n <= 65535)
        .ok_or_else(invalid)? as usize;
    let indices: Vec<u32> = serde_json::from_value(signature["device_indices"].clone())?;
    let uuids: Vec<String> = serde_json::from_value(signature["gpu_uuids"].clone())?;
    if indices.len() != world {
        return Err(invalid());
    }
    let telemetry_value = &data["gpu_telemetry"][batch.to_string()];
    let telemetry: Receipt = serde_json::from_value(telemetry_value.clone())?;
    if telemetry_value["throttled"] != false || !telemetry.verified(&uuids, &indices) {
        return Err(invalid());
    }
    let rows: Vec<_> = data["records"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .filter(|r| r["batch"].as_u64() == Some(batch))
        .collect();
    if rows.len() != world {
        return Err(invalid());
    }
    let parents = rows[0]["parents"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or_else(invalid)?;
    let global = parents.checked_mul(world as u64).ok_or_else(invalid)?;
    let mut seen = BTreeSet::new();
    let mut peak = 0u64;
    let mut repeats = usize::MAX;
    for row in &rows {
        let rank = row["device"]
            .as_u64()
            .filter(|n| *n < (world as u64))
            .ok_or_else(invalid)? as u32;
        if !seen.insert(rank) {
            return Err(invalid());
        }
        validate_receipt(row, rank, batch, parents)?;
        peak = peak.max(row["torch_reserved_peak_bytes"].as_u64().unwrap());
        repeats = repeats.min(row["seconds"].as_array().unwrap().len());
    }
    if data["reserve_bytes"].as_u64() != peak.checked_add(512 << 20) {
        return Err(invalid());
    }
    let measurements: Vec<_> = (0..repeats)
        .map(|i| Measurement {
            profile: batch.to_string(),
            parents: global,
            seconds_by_rank: rows
                .iter()
                .map(|r| r["seconds"][i].as_f64().unwrap())
                .collect(),
            correctness_passed: true,
            memory_admitted: true,
            throttled: false,
        })
        .collect();
    let summary = estimate(&batch.to_string(), &measurements)?;
    let mut normalized = data.clone();
    normalized["estimate"] = serde_json::to_value(summary)?;
    Ok(normalized)
}
fn capacity_reusable(data: &Value, free: Option<&FreeMemory>, signature: &Value) -> bool {
    if data["capacity_limited_search"] != true {
        return true;
    }
    let Some(current) = free else {
        return false;
    };
    let Ok(previous) =
        serde_json::from_value::<FreeMemory>(data["initial_free_memory_mib"].clone())
    else {
        return false;
    };
    let Some(uuids) = signature["gpu_uuids"].as_array() else {
        return false;
    };
    let expected: Option<BTreeSet<_>> = uuids
        .iter()
        .map(|u| {
            u.as_str()
                .filter(|s| !s.is_empty() && *s != "unavailable")
                .map(|s| s.to_ascii_lowercase().trim_start_matches("gpu-").to_owned())
        })
        .collect();
    let Some(expected) = expected else {
        return false;
    };
    if expected.len() != uuids.len()
        || expected.iter().collect::<Vec<_>>() != previous.keys().collect::<Vec<_>>()
        || previous.keys().ne(current.keys())
    {
        return false;
    }
    previous
        .iter()
        .all(|(uuid, before)| current[uuid] <= before.saturating_add(32))
}
pub fn read_inference(
    path: &Path,
    signature: &Value,
    max_batch: u64,
    free: Option<&FreeMemory>,
) -> Result<Option<Value>> {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    const LIMIT: u64 = 64 << 20;
    if file.metadata()?.len() > LIMIT {
        return Ok(None);
    }
    let mut bytes = Vec::new();
    file.by_ref().take(LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > LIMIT {
        return Ok(None);
    }
    let Ok(data) = serde_json::from_slice::<Value>(&bytes) else {
        return Ok(None);
    };
    let Ok(mut data) = normalize(&data, signature, max_batch) else {
        return Ok(None);
    };
    if !capacity_reusable(&data, free, signature) {
        return Ok(None);
    }
    data["cache_hit"] = json!(true);
    Ok(Some(data))
}
pub fn write_inference(path: &Path, data: &Value, signature: &Value, max_batch: u64) -> Result<()> {
    let mut normalized = normalize(data, signature, max_batch)?;
    normalized["cache_hit"] = json!(false);
    artifacts::atomic_json(path, &normalized)
}
