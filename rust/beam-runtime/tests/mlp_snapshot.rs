use beam_runtime::mlp_snapshot::{MlpSnapshotInput, MlpSnapshotStore};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn fixture() -> (PathBuf, MlpSnapshotInput) {
    let root = std::env::temp_dir().join(format!(
        "beam-mlp-snapshot-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let source = root.join("source");
    fs::create_dir_all(&source).unwrap();
    fs::create_dir_all(root.join("cache")).unwrap();
    fs::write(source.join("manifest.json"), b"{\"dtype\":\"fp16\"}\n").unwrap();
    fs::write(source.join("a.fp16"), vec![0x00, 0x3c]).unwrap();
    fs::write(source.join("z.fp16"), vec![0x00; 2 * 65537]).unwrap();
    let graph_hash = "a".repeat(64);
    let tensors = vec!["a.fp16".to_owned(), "z.fp16".to_owned()];
    let artifact_hash = digest(&source, &graph_hash, &tensors);
    let input = MlpSnapshotInput {
        source_dir: source,
        cache_dir: root.join("cache"),
        run_dir: root.join("run1"),
        graph_hash,
        artifact_hash,
        tensors,
        expected_sizes: Some(vec![b"{\"dtype\":\"fp16\"}\n".len() as u64, 2, 2 * 65537]),
    };
    (root, input)
}

fn digest(source: &Path, graph: &str, tensors: &[String]) -> String {
    let mut sha = Sha256::new();
    sha.update(fs::read(source.join("manifest.json")).unwrap());
    sha.update(graph.as_bytes());
    for tensor in tensors {
        sha.update(tensor.as_bytes());
        sha.update(fs::read(source.join(tensor)).unwrap());
    }
    format!("{:x}", sha.finalize())
}

#[test]
fn creates_reuses_and_refreshes_compatible_snapshots() {
    let (root, mut input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let first = store.prepare(&input, false).unwrap();
    assert!(!first.reused);
    assert_eq!(first.path, root.join("run1/weights"));
    assert_eq!(
        first.copied_bytes,
        fs::read(input.source_dir.join("manifest.json"))
            .unwrap()
            .len() as u64
            + 2
            + 2 * 65537
    );
    input.run_dir = root.join("run2");
    let second = store.prepare(&input, false).unwrap();
    assert!(second.reused);
    assert_eq!(second.copied_bytes, 0);
    assert_eq!(second.path, first.path);
    assert!(!input.run_dir.join("weights").exists());
    let third = store.prepare(&input, true).unwrap();
    assert!(!third.reused);
    assert_eq!(third.path, root.join("run2/weights"));
    assert!(first.path.exists());
    assert_eq!(
        fs::read(third.path.join("z.fp16")).unwrap(),
        fs::read(input.source_dir.join("z.fp16")).unwrap()
    );
}

#[test]
fn rejects_changed_source_and_retained_snapshot() {
    let (root, mut input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let first = store.prepare(&input, false).unwrap();
    input.run_dir = root.join("run2");
    fs::write(input.source_dir.join("a.fp16"), [0x01, 0x3c]).unwrap();
    assert!(store.prepare(&input, false).is_err());
    assert!(!input.run_dir.join("weights").exists());
    fs::write(input.source_dir.join("a.fp16"), [0x00, 0x3c]).unwrap();
    fs::write(first.path.join("z.fp16"), [0x00, 0x00]).unwrap();
    assert!(store.prepare(&input, false).is_err());
    assert!(!input.run_dir.join("weights").exists());
}

#[test]
fn launch_verification_accepts_only_the_owned_unmodified_snapshot() {
    let (root, input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let snapshot = store.prepare(&input, false).unwrap();
    let cancel = AtomicBool::new(false);
    let check = |path: &Path, graph: &str, artifact: &str, tensors: &[String]| {
        store.verify_snapshot_with_cancel(path, graph, artifact, tensors, &cancel)
    };
    assert!(check(
        &snapshot.path,
        &input.graph_hash,
        &input.artifact_hash,
        &input.tensors
    )
    .is_ok());
    assert!(check(
        &root.join("other"),
        &input.graph_hash,
        &input.artifact_hash,
        &input.tensors
    )
    .is_err());
    assert!(check(
        &snapshot.path,
        &"b".repeat(64),
        &input.artifact_hash,
        &input.tensors
    )
    .is_err());
    assert!(check(
        &snapshot.path,
        &input.graph_hash,
        &"b".repeat(64),
        &input.tensors
    )
    .is_err());
    assert!(check(
        &snapshot.path,
        &input.graph_hash,
        &input.artifact_hash,
        &input.tensors[..1]
    )
    .is_err());
    fs::write(snapshot.path.join("a.fp16"), [0x01, 0x3c]).unwrap();
    assert!(check(
        &snapshot.path,
        &input.graph_hash,
        &input.artifact_hash,
        &input.tensors
    )
    .is_err());
}

#[test]
fn launch_verification_honors_cancellation_and_clear() {
    let (_, input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let snapshot = store.prepare(&input, false).unwrap();
    let cancelled = AtomicBool::new(true);
    assert!(store
        .verify_snapshot_with_cancel(
            &snapshot.path,
            &input.graph_hash,
            &input.artifact_hash,
            &input.tensors,
            &cancelled,
        )
        .is_err());
    store.clear();
    assert!(store
        .verify_snapshot_with_cancel(
            &snapshot.path,
            &input.graph_hash,
            &input.artifact_hash,
            &input.tensors,
            &AtomicBool::new(false),
        )
        .is_err());
}

#[test]
fn launch_verification_rejects_identical_digest_with_shifted_file_boundaries() {
    let (_, mut input) = fixture();
    let a = [b"\x00\x3c".as_slice(), b"z.fp16", b"\x00\x3c"].concat();
    let b = b"\x00\x3c\x00\x3c";
    fs::write(input.source_dir.join("a.fp16"), a).unwrap();
    fs::write(input.source_dir.join("z.fp16"), b).unwrap();
    input.expected_sizes.as_mut().unwrap()[1..].copy_from_slice(&[10, 4]);
    input.artifact_hash = digest(&input.source_dir, &input.graph_hash, &input.tensors);
    let mut store = MlpSnapshotStore::default();
    let snapshot = store.prepare(&input, false).unwrap();
    fs::write(snapshot.path.join("a.fp16"), b"\x00\x3c").unwrap();
    fs::write(
        snapshot.path.join("z.fp16"),
        [b"\x00\x3c".as_slice(), b"z.fp16", b].concat(),
    )
    .unwrap();
    assert_eq!(
        digest(&snapshot.path, &input.graph_hash, &input.tensors),
        input.artifact_hash
    );
    assert!(store
        .verify_snapshot_with_cancel(
            &snapshot.path,
            &input.graph_hash,
            &input.artifact_hash,
            &input.tensors,
            &AtomicBool::new(false),
        )
        .is_err());
}

#[test]
fn copying_rejects_shifted_boundaries_even_when_digest_is_unchanged() {
    let (_, mut input) = fixture();
    let a = [b"\x00\x3c".as_slice(), b"z.fp16", b"\x00\x3c"].concat();
    let b = b"\x00\x3c\x00\x3c";
    fs::write(input.source_dir.join("a.fp16"), a).unwrap();
    fs::write(input.source_dir.join("z.fp16"), b).unwrap();
    input.expected_sizes.as_mut().unwrap()[1..].copy_from_slice(&[10, 4]);
    input.artifact_hash = digest(&input.source_dir, &input.graph_hash, &input.tensors);
    fs::write(input.source_dir.join("a.fp16"), b"\x00\x3c").unwrap();
    fs::write(
        input.source_dir.join("z.fp16"),
        [b"\x00\x3c".as_slice(), b"z.fp16", b].concat(),
    )
    .unwrap();
    assert_eq!(
        digest(&input.source_dir, &input.graph_hash, &input.tensors),
        input.artifact_hash
    );
    assert!(MlpSnapshotStore::default().prepare(&input, false).is_err());
    assert!(!input.run_dir.join("weights").exists());
}

#[test]
fn verification_rejects_manifest_boundary_shift_with_identical_digest() {
    let (_, mut input) = fixture();
    let prefix = b"{\"note\":\"";
    let suffix = b"\"}\n";
    let manifest = [
        prefix.as_slice(),
        input.graph_hash.as_bytes(),
        b"a.fp16",
        suffix,
    ]
    .concat();
    let tensor = fs::read(input.source_dir.join("a.fp16")).unwrap();
    fs::write(input.source_dir.join("manifest.json"), &manifest).unwrap();
    input.expected_sizes.as_mut().unwrap()[0] = manifest.len() as u64;
    input.artifact_hash = digest(&input.source_dir, &input.graph_hash, &input.tensors);
    let mut store = MlpSnapshotStore::default();
    let snapshot = store.prepare(&input, false).unwrap();
    fs::write(snapshot.path.join("manifest.json"), prefix).unwrap();
    fs::write(
        snapshot.path.join("a.fp16"),
        [
            suffix.as_slice(),
            input.graph_hash.as_bytes(),
            b"a.fp16",
            &tensor,
        ]
        .concat(),
    )
    .unwrap();
    assert_eq!(
        digest(&snapshot.path, &input.graph_hash, &input.tensors),
        input.artifact_hash
    );
    assert!(store
        .verify_snapshot_with_cancel(
            &snapshot.path,
            &input.graph_hash,
            &input.artifact_hash,
            &input.tensors,
            &AtomicBool::new(false),
        )
        .is_err());
}

#[test]
fn unsized_legacy_snapshot_cannot_use_fast_launch_verification() {
    let (_, mut input) = fixture();
    input.expected_sizes = None;
    let mut store = MlpSnapshotStore::default();
    let snapshot = store.prepare(&input, false).unwrap();
    assert!(store
        .verify_snapshot_with_cancel(
            &snapshot.path,
            &input.graph_hash,
            &input.artifact_hash,
            &input.tensors,
            &AtomicBool::new(false),
        )
        .is_err());
    input.run_dir = input.run_dir.with_file_name("run2");
    input.expected_sizes = Some(vec![1]);
    assert!(store.prepare(&input, false).is_err());
}

#[test]
fn session_path_defers_retained_validation_until_the_launch_boundary() {
    let (root, mut input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let first = store.prepare(&input, false).unwrap();
    fs::write(first.path.join("z.fp16"), [0x00, 0x00]).unwrap();
    input.run_dir = root.join("run2");
    let cancel = AtomicBool::new(false);
    let deferred = store
        .prepare_for_session_with_cancel(&input, false, &cancel)
        .unwrap();
    assert!(deferred.reused);
    assert!(store.prepare(&input, false).is_err());
}

#[test]
fn rejects_bad_hash_before_publishing_and_preserves_existing_target() {
    let (root, mut input) = fixture();
    let mut store = MlpSnapshotStore::default();
    input.artifact_hash = "b".repeat(64);
    assert!(store.prepare(&input, false).is_err());
    assert!(!input.run_dir.join("weights").exists());
    let leftovers: Vec<_> = fs::read_dir(&input.run_dir)
        .unwrap()
        .filter_map(|entry| {
            let name = entry.unwrap().file_name();
            name.to_str()
                .filter(|name| name.starts_with(".weights-snapshot-"))
                .map(str::to_owned)
        })
        .collect();
    assert!(leftovers.is_empty());
    fs::create_dir_all(input.run_dir.join("weights")).unwrap();
    fs::write(input.run_dir.join("weights/user-file"), b"keep").unwrap();
    input.artifact_hash = digest(&input.source_dir, &input.graph_hash, &input.tensors);
    assert!(store.prepare(&input, false).is_err());
    assert_eq!(
        fs::read(root.join("run1/weights/user-file")).unwrap(),
        b"keep"
    );
}

#[test]
fn cancellation_does_not_publish_or_poison_later_attempts() {
    let (_, input) = fixture();
    let mut store = MlpSnapshotStore::default();
    let cancelled = AtomicBool::new(true);
    assert!(store
        .prepare_with_cancel(&input, false, &cancelled)
        .is_err());
    assert!(!input.run_dir.join("weights").exists());
    assert!(!store.prepare(&input, false).unwrap().reused);
}

#[test]
fn cache_identity_need_not_exist_when_the_run_directory_is_elsewhere() {
    let (root, mut input) = fixture();
    input.cache_dir = root.join("not-yet-created-cache");
    let snapshot = MlpSnapshotStore::default().prepare(&input, false).unwrap();
    assert!(snapshot.path.is_dir());
    assert!(!input.cache_dir.exists());
}

#[test]
fn already_prepared_source_at_target_is_reused_without_a_copy() {
    let (root, mut input) = fixture();
    input.run_dir = root.clone();
    let target = root.join("weights");
    fs::rename(&input.source_dir, &target).unwrap();
    input.source_dir = target.clone();
    let snapshot = MlpSnapshotStore::default().prepare(&input, false).unwrap();
    assert_eq!(snapshot.path, target);
    assert_eq!(snapshot.copied_bytes, 0);
    assert!(!snapshot.reused);
}

#[cfg(unix)]
#[test]
fn fifo_tensor_cannot_block_snapshot_or_cancellation() {
    use std::{
        process::Command,
        sync::{mpsc, Arc},
        thread,
        time::Duration,
    };
    let (_, input) = fixture();
    let path = input.source_dir.join("a.fp16");
    fs::remove_file(&path).unwrap();
    assert!(Command::new("mkfifo")
        .arg(&path)
        .status()
        .unwrap()
        .success());
    let cancel = Arc::new(AtomicBool::new(false));
    let worker_cancel = Arc::clone(&cancel);
    let (send, receive) = mpsc::channel();
    thread::spawn(move || {
        let result = MlpSnapshotStore::default().prepare_with_cancel(&input, false, &worker_cancel);
        let _ = send.send(result);
    });
    thread::sleep(Duration::from_millis(50));
    cancel.store(true, Ordering::Release);
    assert!(receive
        .recv_timeout(Duration::from_secs(2))
        .expect("snapshot was blocked opening a FIFO despite cancellation")
        .is_err());
}

#[test]
fn rejects_invalid_or_ambiguous_file_names_and_hashes() {
    let (_, input) = fixture();
    for tensors in [
        vec!["../a.fp16".to_owned()],
        vec!["a.fp16".to_owned(), "a.fp16".to_owned()],
        vec!["z.fp16".to_owned(), "a.fp16".to_owned()],
        vec!["manifest.json".to_owned()],
        vec![],
    ] {
        let mut bad = input.clone();
        bad.tensors = tensors;
        assert!(MlpSnapshotStore::default().prepare(&bad, false).is_err());
    }
    for graph in ["a".repeat(63), "A".repeat(64), "z".repeat(64)] {
        let mut bad = input.clone();
        bad.graph_hash = graph;
        assert!(MlpSnapshotStore::default().prepare(&bad, false).is_err());
    }
}
