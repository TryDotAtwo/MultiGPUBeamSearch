use beam_runtime::inference::{InferencePool, PoolConfig, Protocol};
use std::{collections::BTreeMap, fs, time::Duration};

#[test]
fn inference_starts_all_ranks_before_requests_and_restarts_only_failed_rank() {
    let root = std::env::temp_dir().join(format!("beam-inference-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let worker = root.join("worker.py");
    fs::write(
        &worker,
        r#"import sys,json,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(sys.argv[4]);(root/f'started-{rank}').touch()
while not all((root/f'started-{r}').exists() for r in range(2)):time.sleep(.001)
print(json.dumps(dict(ready=True,device=rank)),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 time.sleep(.02)
 if r['batch']==16 and rank==1:raise SystemExit(7)
 print(json.dumps(dict(batch=r['batch'],parents=r['parents'],device=rank,
  correctness_passed=True,numeric_error=0,seconds=[.1]*7,torch_reserved_peak_bytes=0)),flush=True)
"#,
    )
    .unwrap();
    let config = PoolConfig {
        command: vec!["python".into(), worker.to_string_lossy().into_owned()],
        inputs: root.clone(),
        capacity: 32,
        parents: 64,
        world: 2,
        environment: std::env::vars().collect::<BTreeMap<_, _>>(),
        cwd: root.clone(),
        directory: root.join("logs"),
        protocol: Protocol::Session,
    };
    let mut pool = InferencePool::new(config, Duration::from_secs(20)).unwrap();
    let first = pool.measure(8, 64, Duration::from_secs(1)).unwrap();
    assert!(first.failures.is_empty());
    assert_eq!(first.rows.len(), 2);
    assert_eq!(pool.starts(), 2);
    let pids = pool.pids();
    // A measurement's deadline must not expire the retained worker itself.
    std::thread::sleep(Duration::from_millis(1100));
    let failed = pool.measure(16, 64, Duration::from_secs(5)).unwrap();
    assert_eq!(failed.failures.len(), 1);
    assert_eq!(failed.failures[0].rank, 1);
    assert!(!failed.failures[0].capacity);
    let again = pool.measure(4, 64, Duration::from_secs(5)).unwrap();
    assert!(again.failures.is_empty());
    assert_eq!(pool.starts(), 3);
    assert_eq!(pool.pids()[0], pids[0]);
    assert_ne!(pool.pids()[1], pids[1]);
    pool.close().unwrap();
    assert!(pool.pids().iter().all(Option::is_none));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn one_shot_inference_validates_receipts_and_reaps_helpers() {
    let root = std::env::temp_dir().join(format!("beam-inference-once-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let worker = root.join("worker.py");
    fs::write(
        &worker,
        r#"import sys,json
batch,parents,rank=map(int,sys.argv[2:5])
print(json.dumps(dict(batch=batch,parents=parents,device=rank,correctness_passed=batch!=16,
 numeric_error=0,seconds=[.1]*7,torch_reserved_peak_bytes=1024)),flush=True)
"#,
    )
    .unwrap();
    let config = PoolConfig {
        command: vec!["python".into(), worker.to_string_lossy().into_owned()],
        inputs: root.clone(),
        capacity: 32,
        parents: 64,
        world: 2,
        environment: std::env::vars().collect(),
        cwd: root.clone(),
        directory: root.join("logs"),
        protocol: Protocol::Oneshot,
    };
    let mut pool = InferencePool::new(config, Duration::from_secs(20)).unwrap();
    assert_eq!(
        pool.measure(8, 64, Duration::from_secs(5))
            .unwrap()
            .rows
            .len(),
        2
    );
    assert!(pool.pids().iter().all(Option::is_none));
    let bad = pool.measure(16, 64, Duration::from_secs(5)).unwrap();
    assert_eq!(bad.failures.len(), 2);
    assert!(bad.rows.is_empty());
    assert!(pool.pids().iter().all(Option::is_none));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn readiness_oom_reduces_workspace_but_bad_quality_receipt_is_not_capacity_evidence() {
    let root = std::env::temp_dir().join(format!("beam-inference-capacity-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let worker = root.join("worker.py");
    fs::write(&worker,r#"import sys,json
capacity=int(sys.argv[2]);rank=int(sys.argv[4])
if capacity>8:
 print('CUDA out of memory',file=sys.stderr,flush=True);raise SystemExit(2)
print(json.dumps(dict(ready=True,device=rank)),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r['batch']==8:print('previous diagnostic: CUDA out of memory',file=sys.stderr,flush=True)
 print(json.dumps(dict(batch=r['batch'],parents=r['parents'],device=rank,
  correctness_passed=r['batch']!=8,numeric_error=0,seconds=[.1]*7,torch_reserved_peak_bytes=0)),flush=True)
"#).unwrap();
    let config = PoolConfig {
        command: vec!["python".into(), worker.to_string_lossy().into_owned()],
        inputs: root.clone(),
        capacity: 32,
        parents: 64,
        world: 2,
        environment: std::env::vars().collect(),
        cwd: root.clone(),
        directory: root.join("logs"),
        protocol: Protocol::Session,
    };
    let mut pool = InferencePool::new(config, Duration::from_secs(20)).unwrap();
    let oom = pool.measure(8, 64, Duration::from_secs(5)).unwrap();
    assert!(oom.failures.iter().all(|failure| failure.capacity));
    assert!(!oom.failures.is_empty());
    assert_eq!(pool.capacity(), 8);
    assert_eq!(pool.pids(), vec![None, None]);
    let good = pool.measure(4, 64, Duration::from_secs(5)).unwrap();
    assert_eq!(good.rows.len(), 2);
    let bad = pool.measure(8, 64, Duration::from_secs(5)).unwrap();
    assert_eq!(bad.failures.len(), 2);
    assert!(bad.failures.iter().all(|failure| !failure.capacity));
    assert_eq!(pool.pids(), vec![None, None]);
    pool.close().unwrap();
    fs::remove_dir_all(root).unwrap();
}
