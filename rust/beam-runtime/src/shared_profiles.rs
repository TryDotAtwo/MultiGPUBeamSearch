//! Fail-closed lookup of hash-addressed shared inference profile proposals.
use serde_json::Value;
use std::{
    fs::{self, File},
    io::Read,
    path::Path,
};

const SCHEMA: u64 = 1;
const MAX_BYTES: u64 = 2 << 20;
const MAX_PROFILES: usize = 1024;
const MAX_PARENT_BATCH: u64 = 65_536;

fn read_limited(path: &Path) -> Option<Vec<u8>> {
    let file = File::open(path).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_BYTES {
        return None;
    }
    let mut bytes = Vec::new();
    file.take(MAX_BYTES + 1).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 > MAX_BYTES {
        return None;
    }
    Some(bytes)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn index_bytes(profiles_root: &Path) -> Option<Value> {
    let bytes = read_limited(&profiles_root.join("index.json"))?;
    serde_json::from_slice(&bytes).ok()
}

/// Returns the verified proposal selected by a source-pinned profile index.
/// Every filesystem, schema, path and digest failure is a cache miss.
pub fn find_profile(profiles_root: &Path, identity: &Value) -> Option<Value> {
    let root = profiles_root.canonicalize().ok()?;
    if !root.is_dir() {
        return None;
    }
    let index = index_bytes(&root)?;
    if index.get("schema").and_then(Value::as_u64) != Some(SCHEMA) {
        return None;
    }
    let entries = index.get("profiles")?.as_array()?;
    if entries.len() > MAX_PROFILES {
        return None;
    }

    for entry in entries {
        let fields = entry.as_object()?;
        if fields.get("compatibility") != Some(identity) {
            continue;
        }
        let digest_value = fields.get("sha256")?;
        let Some(digest) = digest_value.as_str() else {
            continue;
        };
        if !valid_sha256(digest) {
            continue;
        }

        let record_path = root.join("records").join(format!("{digest}.json"));
        let metadata = match fs::symlink_metadata(&record_path) {
            Ok(metadata) if metadata.file_type().is_symlink() => continue,
            Ok(metadata) => metadata,
            Err(_) => return None,
        };
        if metadata.len() > MAX_BYTES {
            continue;
        }
        let resolved = record_path.canonicalize().ok()?;
        if !resolved.starts_with(&root) {
            continue;
        }
        let payload = read_limited(&record_path)?;
        if crate::artifacts::sha256_bytes(&payload) != digest {
            continue;
        }
        let mut record: Value = serde_json::from_slice(&payload).ok()?;
        let fields = record.as_object_mut()?;
        if fields.get("schema").and_then(Value::as_u64) != Some(SCHEMA)
            || fields.get("compatibility") != Some(identity)
        {
            continue;
        }
        let Some(evidence_value) = fields.get("evidence") else {
            continue;
        };
        let evidence = evidence_value.as_object()?;
        if evidence.get("status").and_then(Value::as_str) != Some("verified")
            || evidence
                .get("source_commit")
                .and_then(Value::as_str)
                .is_none()
        {
            continue;
        }
        let Some(inference_value) = fields.get("inference") else {
            continue;
        };
        let inference = inference_value.as_object()?;
        if !matches!(
            inference.get("parent_batch").and_then(Value::as_u64),
            Some(1..=MAX_PARENT_BATCH)
        ) {
            continue;
        }
        fields.insert("registry_sha256".into(), Value::String(digest.to_owned()));
        return Some(record);
    }
    None
}
