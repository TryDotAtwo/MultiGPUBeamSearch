use beam_runtime::ResidentWorkers;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "beam-runtime-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn workers(root: &Temporary, mode: &str) -> ResidentWorkers {
    let script = r#"
import os,json,time,pathlib,sys
d=pathlib.Path(os.environ['BEAM_SESSION_DIR']);r=int(os.environ['RANK']);n=0
while not (d/'stop').exists():
 p=d/f'request-{n}.json'
 if not p.exists():time.sleep(.005);continue
 if sys.argv[1]=='fail' and r==1:sys.exit(19)
 if sys.argv[1]=='stall':time.sleep(30)
 q=json.loads(p.read_text());out=pathlib.Path(q['run_dir'])
 (out/f'resident-rank-{r}.log').write_text(f'sequence={n} state={q["start"]}')
 p=d/f'done-{n}-{r}.json';t=p.with_suffix('.tmp')
 t.write_text(json.dumps(dict(sequence=n,rank=r)));t.replace(p);n+=1
"#;
    let env: BTreeMap<String, String> = std::env::vars().collect();
    ResidentWorkers::launch(
        &["python3".into(), "-c".into(), script.into(), mode.into()],
        2,
        &root.0.join("session"),
        &root.0,
        &env,
    )
    .unwrap()
}
#[test]
fn repeated_jobs_keep_rank_pids_and_merge_logs() {
    let root = Temporary::new();
    let mut group = workers(&root, "ok");
    let pids = group.pids();
    for n in 0..2 {
        let run = root.0.join(format!("run-{n}"));
        fs::create_dir(&run).unwrap();
        group
            .submit(&[1, 2, 3], &run, Duration::from_secs(5))
            .unwrap();
        let log = fs::read_to_string(run.join("native.log")).unwrap();
        assert!(log.contains("[rank 0]"));
        assert!(log.contains("[rank 1]"));
        assert!(log.contains(&format!("sequence={n}")));
        assert_eq!(group.pids(), pids);
    }
    assert_eq!(group.sequence(), 2);
    group.close().unwrap();
    assert!(group.is_closed());
    assert!(group.submit(&[1], &root.0, Duration::from_secs(1)).is_err());
}
#[test]
fn a_failed_rank_closes_the_entire_owned_group() {
    let root = Temporary::new();
    let mut group = workers(&root, "fail");
    let error = group
        .submit(&[1], &root.0, Duration::from_secs(5))
        .unwrap_err();
    assert!(error.to_string().contains("exited"));
    assert!(group.is_closed());
}
#[test]
fn timeout_and_cancellation_are_terminal() {
    let root = Temporary::new();
    let mut group = workers(&root, "stall");
    let error = group
        .submit(&[1], &root.0, Duration::from_millis(100))
        .unwrap_err();
    assert!(error.to_string().contains("timed out"));
    assert!(group.is_closed());
    let second = Temporary::new();
    let mut group = workers(&second, "ok");
    group
        .cancellation()
        .store(true, std::sync::atomic::Ordering::Release);
    assert!(group
        .submit(&[1], &second.0, Duration::from_secs(1))
        .is_err());
    assert!(group.is_closed());
}
