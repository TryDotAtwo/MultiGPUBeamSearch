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
