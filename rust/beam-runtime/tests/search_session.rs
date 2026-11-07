use beam_core::search::PermutationGraph;
use beam_runtime::search_session::{Launch, SearchSession, SessionIdentity};
use serde_json::json;
use std::{fs, time::Duration};
#[test]
fn session_reuses_compatible_jobs_and_releases_before_model_or_geometry_change() {
    let root = std::env::temp_dir().join(format!("beam-search-owner-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let worker = root.join("worker.py");
    fs::write(
        &worker,
        r#"import sys,os,json,time
from pathlib import Path
if '--resident-protocol-info' in sys.argv:
 print('{"protocol":"resident-search-v1"}',flush=True);raise SystemExit(0)
directory=Path(os.environ['BEAM_SESSION_DIR']);rank=int(os.environ['RANK']);sequence=0
while not (directory/'stop').exists():
 path=directory/f'request-{sequence}.json'
 if not path.exists():time.sleep(.002);continue
 request=json.loads(path.read_text());start=request['start'];out=Path(request['run_dir'])
 moves='m0' if start==[1,0,2,3] else 'm1' if start==[0,1,3,2] else None
 text='GLOBAL_BEAM_WIDTH_EFFECTIVE='+os.environ['FIXTURE_BEAM']+'\n'
 if moves is None:text+='puzzle_solved=0 puzzle_id=0 seconds=.1 solution_length=-1 solution=\n'
 elif rank==0:text+='puzzle_solved=1 puzzle_id=0 seconds=.1 solution_length=1 solution='+moves+'\n'
 (out/f'resident-rank-{rank}.log').write_text(text)
 path=directory/f'done-{sequence}-{rank}.json';temp=path.with_suffix('.tmp')
 temp.write_text(json.dumps(dict(sequence=sequence,rank=rank)));temp.replace(path);sequence+=1
"#,
    )
    .unwrap();
    let mut identity = SessionIdentity {
        graph: PermutationGraph {
            center: vec![0, 1, 2, 3],
            generators: vec![vec![1, 0, 2, 3], vec![0, 1, 3, 2]],
        },
        graph_hash: "a".repeat(64),
        model_artifact_hash: "b".repeat(64),
        model_backend: "mlp".into(),
        runtime_build: json!({"binary_sha256":"c".repeat(64)}),
        devices: vec![0, 1],
        beam_width: 64,
        max_steps: 1,
        options: json!({"solution_mode":"first"}),
        cuda_visible_devices: None,
        cuda_device_order: None,
    };
    let command = vec!["python".into(), worker.to_string_lossy().into_owned()];
    let mut env: std::collections::BTreeMap<_, _> = std::env::vars().collect();
    env.insert("FIXTURE_BEAM".into(), "64".into());
    let mut launch = Launch {
        command: command.clone(),
        protocol_command: [command, vec!["--resident-protocol-info".into()]].concat(),
        environment: env,
        cwd: root.clone(),
    };
    let metadata = json!({"profile":{"calibration_total_wall_seconds":12.0},"model_artifact_hash":identity.model_artifact_hash});
    let mut session = SearchSession::new();
    let mut previous = Vec::new();
    for (index, start) in [vec![1, 0, 2, 3], vec![0, 1, 3, 2], vec![1, 0, 3, 2]]
        .iter()
        .enumerate()
    {
        let dir = root.join(format!("run-{index}"));
        fs::create_dir(&dir).unwrap();
        let result = session
            .run(
                &identity,
                &launch,
                start,
                &dir,
                Duration::from_secs(5),
                &metadata,
            )
            .unwrap();
        assert_eq!(result.effective_beam_width, Some(64));
        assert_eq!(result.metadata["persistent_session"]["reused"], index > 0);
        if index > 0 {
            assert_eq!(session.pids(), previous);
            assert_eq!(
                result.metadata["profile"]["calibration_total_wall_seconds"],
                0.0
            );
        }
        previous = session.pids();
        assert_eq!(
            result.path,
            if index < 2 {
                Some(vec![index as u32])
            } else {
                None
            }
        );
    }
    identity.model_artifact_hash = "d".repeat(64);
    let dir = root.join("model-change");
    fs::create_dir(&dir).unwrap();
    let changed = session
        .run(
            &identity,
            &launch,
            &[1, 0, 2, 3],
            &dir,
            Duration::from_secs(5),
            &metadata,
        )
        .unwrap();
    assert_eq!(changed.metadata["persistent_session"]["reused"], false);
    assert_ne!(session.pids(), previous);
    #[cfg(unix)]
    for pid in previous {
        assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists());
    }
    previous = session.pids();
    identity.beam_width = 65;
    launch
        .environment
        .insert("FIXTURE_BEAM".into(), "65".into());
    let dir = root.join("geometry-change");
    fs::create_dir(&dir).unwrap();
    assert_eq!(
        session
            .run(
                &identity,
                &launch,
                &[1, 0, 2, 3],
                &dir,
                Duration::from_secs(5),
                &metadata
            )
            .unwrap()
            .effective_beam_width,
        Some(65)
    );
    assert_ne!(session.pids(), previous);
    session.close().unwrap();
    assert!(session.pids().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn session_timeout_includes_protocol_preparation_and_reaps_delayed_workers() {
    let root = std::env::temp_dir().join(format!("beam-search-deadline-{}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let worker = root.join("worker.py");
    fs::write(&worker,r#"import os,sys,json,time
from pathlib import Path
if '--resident-protocol-info' in sys.argv:
 time.sleep(.3);print('{"protocol":"resident-search-v1"}',flush=True);raise SystemExit(0)
directory=Path(os.environ['BEAM_SESSION_DIR'])
while not (directory/'stop').exists():
 path=directory/'request-0.json'
 if not path.exists():time.sleep(.002);continue
 time.sleep(.3);request=json.loads(path.read_text());out=Path(request['run_dir'])
 (out/'resident-rank-0.log').write_text('GLOBAL_BEAM_WIDTH_EFFECTIVE=64\npuzzle_solved=1 puzzle_id=0 seconds=.3 solution_length=1 solution=m0\n')
 (directory/'done-0-0.json').write_text(json.dumps(dict(sequence=0,rank=0)))
 while not (directory/'stop').exists():time.sleep(.002)
"#).unwrap();
    let identity = SessionIdentity {
        graph: PermutationGraph {
            center: vec![0, 1],
            generators: vec![vec![1, 0]],
        },
        graph_hash: "a".repeat(64),
        model_artifact_hash: "b".repeat(64),
        model_backend: "mlp".into(),
        runtime_build: json!({"binary_sha256":"c".repeat(64)}),
        devices: vec![0],
        beam_width: 64,
        max_steps: 1,
        options: json!({}),
        cuda_visible_devices: None,
        cuda_device_order: None,
    };
    let command = vec!["python".into(), worker.to_string_lossy().into_owned()];
    let launch = Launch {
        command: command.clone(),
        protocol_command: [command, vec!["--resident-protocol-info".into()]].concat(),
        environment: std::env::vars().collect(),
        cwd: root.clone(),
    };
    let mut owner = SearchSession::new();
    let result = owner.run(
        &identity,
        &launch,
        &[1, 0],
        &root,
        Duration::from_millis(500),
        &json!({}),
    );
    assert!(
        result.is_err(),
        "request deadline was restarted after protocol preparation"
    );
    assert!(owner.pids().is_empty());
    owner.close().unwrap();
    fs::remove_dir_all(root).unwrap();
}
