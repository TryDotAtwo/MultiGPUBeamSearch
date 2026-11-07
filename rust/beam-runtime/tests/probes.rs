use beam_runtime::probes::{ProbeCohort, ProbeSession};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

fn fixture(name: &str) -> (PathBuf, Vec<String>, BTreeMap<String, String>) {
    let dir = std::env::temp_dir().join(format!("beam-probe-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let worker = dir.join("worker.py");
    fs::write(
        &worker,
        r#"import sys,json,time
print('diagnostic line',flush=True)
print(json.dumps({'ready':True}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request.get('exit'): sys.exit(7)
    time.sleep(request.get('delay',0))
    print(json.dumps({'echo':request}),flush=True)
"#,
    )
    .unwrap();
    let env = std::env::vars().collect();
    (
        dir,
        vec!["python".into(), worker.to_string_lossy().into()],
        env,
    )
}
#[test]
fn retained_probe_ignores_logs_and_reuses_process() {
    let (dir, command, env) = fixture("repeat");
    let mut session = ProbeSession::launch(
        &command,
        &dir,
        &env,
        &dir.join("probe.log"),
        Duration::from_secs(5),
    )
    .unwrap();
    let pid = session.pid();
    assert_eq!(
        session.receive(Duration::from_secs(2)).unwrap(),
        json!({"ready":true})
    );
    for batch in [1, 8, 3] {
        session.send(&json!({"batch":batch})).unwrap();
        assert_eq!(
            session.receive(Duration::from_secs(2)).unwrap(),
            json!({"echo":{"batch":batch}})
        );
        assert_eq!(session.pid(), pid);
    }
    session.close().unwrap();
    assert!(session.exit_code().unwrap().is_some());
    assert!(fs::read_to_string(dir.join("probe.log"))
        .unwrap()
        .contains("diagnostic line"));
    fs::remove_dir_all(dir).unwrap();
}
#[test]
fn deadline_and_one_rank_failure_close_owned_cohort() {
    let (dir, command, env) = fixture("failure");
    let mut group = ProbeCohort::launch(
        &[command.clone(), command.clone()],
        &[env.clone(), env.clone()],
        &dir,
        &dir.join("cohort"),
        Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(group.collect(Duration::from_secs(2)).unwrap().len(), 2);
    group
        .send(&[json!({"delay":30}), json!({"exit":true})])
        .unwrap();
    let began = Instant::now();
    assert!(group.collect(Duration::from_secs(4)).is_err());
    assert!(began.elapsed() < Duration::from_secs(3));
    assert!(group.exit_codes().unwrap().iter().all(Option::is_some));
    let mut session = ProbeSession::launch(
        &command,
        &dir,
        &env,
        &dir.join("timeout.log"),
        Duration::from_secs(3),
    )
    .unwrap();
    session.receive(Duration::from_secs(2)).unwrap();
    session.send(&json!({"delay":30})).unwrap();
    assert!(session
        .receive(Duration::from_millis(10))
        .unwrap_err()
        .to_string()
        .contains("timed out"));
    session.close().unwrap();
    assert!(session.exit_code().unwrap().is_some());
    fs::remove_dir_all(dir).unwrap();
}
