use beam_runtime::{artifacts::sha256_bytes, shared_profiles::find_profile};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "beam-shared-profile-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn identity() -> Value {
    json!({"backend": "cutlass", "beam": 1_000_000, "world_size": 2})
}

fn record(identity: &Value) -> Value {
    json!({
        "schema": 1,
        "compatibility": identity,
        "inference": {"parent_batch": 512},
        "evidence": {"status": "verified", "source_commit": "a".repeat(40)}
    })
}

fn write_index(root: &Path, entries: Value) {
    fs::create_dir_all(root.join("records")).unwrap();
    fs::write(
        root.join("index.json"),
        serde_json::to_vec(&json!({"schema": 1, "profiles": entries})).unwrap(),
    )
    .unwrap();
}

fn write_record(root: &Path, identity: &Value, value: &Value) -> String {
    fs::create_dir_all(root.join("records")).unwrap();
    let payload = serde_json::to_vec(value).unwrap();
    let digest = sha256_bytes(&payload);
    fs::write(root.join("records").join(format!("{digest}.json")), &payload).unwrap();
    write_index(
        root,
        json!([{"compatibility": identity, "sha256": digest}]),
    );
    digest
}

#[test]
fn accepts_only_a_matching_hash_addressed_verified_proposal() {
    let scratch = Scratch::new();
    let root = scratch.path().join("profiles");
    let identity = identity();
    let digest = write_record(&root, &identity, &record(&identity));

    let selected = find_profile(&root, &identity).unwrap();
    assert_eq!(selected["registry_sha256"].as_str(), Some(digest.as_str()));
    assert_eq!(selected["inference"]["parent_batch"], 512);
}

#[test]
fn rejects_workload_mismatch_and_tampered_record_bytes() {
    let scratch = Scratch::new();
    let root = scratch.path().join("profiles");
    let identity = identity();
    let digest = write_record(&root, &identity, &record(&identity));

    assert!(
        find_profile(
            &root,
            &json!({"backend": "cutlass", "beam": 8, "world_size": 2})
        )
        .is_none()
    );
    fs::write(root.join("records").join(format!("{digest}.json")), b"{}").unwrap();
    assert!(find_profile(&root, &identity).is_none());
}

#[test]
fn rejects_oversized_index_and_more_than_the_entry_limit() {
    let scratch = Scratch::new();
    let root = scratch.path().join("profiles");
    fs::create_dir_all(root.join("records")).unwrap();
    fs::write(root.join("index.json"), vec![b' '; (2 << 20) + 1]).unwrap();
    assert!(find_profile(&root, &identity()).is_none());

    let entries = vec![json!({"compatibility": null, "sha256": "0".repeat(64)}); 1025];
    write_index(&root, json!(entries));
    assert!(find_profile(&root, &identity()).is_none());
}

#[cfg(unix)]
#[test]
fn rejects_record_symlinks_and_resolved_paths_outside_the_registry() {
    let scratch = Scratch::new();
    let root = scratch.path().join("profiles");
    let identity = identity();
    fs::create_dir_all(root.join("records")).unwrap();
    let external = scratch.path().join("external");
    fs::create_dir_all(&external).unwrap();
    let value = record(&identity);
    let payload = serde_json::to_vec(&value).unwrap();
    let digest = sha256_bytes(&payload);
    let outside_record = external.join(format!("{digest}.json"));
    fs::write(&outside_record, payload).unwrap();
    std::os::unix::fs::symlink(
        &outside_record,
        root.join("records").join(format!("{digest}.json")),
    )
    .unwrap();
    write_index(
        &root,
        json!([{"compatibility": identity, "sha256": digest}]),
    );
    assert!(find_profile(&root, &value["compatibility"]).is_none());

    fs::remove_file(root.join("records").join(format!("{digest}.json")).as_path()).unwrap();
    fs::remove_dir(root.join("records")).unwrap();
    std::os::unix::fs::symlink(&external, root.join("records")).unwrap();
    assert!(find_profile(&root, &value["compatibility"]).is_none());
}
