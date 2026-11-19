use beam_core::search::PermutationGraph;
use beam_runtime::search_session::{Launch, PreparationIdentity, SearchSession, SessionIdentity};
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
 reply=dict(protocol='resident-search-v1')
 if os.environ.get('BEAM_REQUIRE_LOADED_GRAPH_BINDING')=='1' and os.environ.get('FIXTURE_LEGACY_PROBE')!='1':reply['loaded_graph_binding']=1
 print(json.dumps(reply),flush=True);raise SystemExit(0)
directory=Path(os.environ['BEAM_SESSION_DIR']);rank=int(os.environ['RANK']);sequence=0
assert os.environ['BEAM_EXPECTED_GRAPH_HASH']=='0dfd25a031f018028e013a6cbbf83b3588c61b5b26897bb5321d117d7d3cca8d'
while not (directory/'stop').exists():
 path=directory/f'request-{sequence}.json'
 if not path.exists():time.sleep(.002);continue
 request=json.loads(path.read_text());start=request['start'];out=Path(request['run_dir'])
 moves='m0' if start==[1,0,2,3] else 'm1' if start==[0,1,3,2] else None
 text='GLOBAL_BEAM_WIDTH_EFFECTIVE='+os.environ['FIXTURE_BEAM']+'\n'
 text+='B_MICRO='+str(32 if sequence>0 or rank==0 else 64)+'\n'
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
        graph_hash: "0dfd25a031f018028e013a6cbbf83b3588c61b5b26897bb5321d117d7d3cca8d".into(),
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
    env.insert("BEAM_EXPECTED_GRAPH_HASH".into(), "caller-spoof".into());
    env.insert("BEAM_REQUIRE_LOADED_GRAPH_BINDING".into(), "0".into());
    let mut launch = Launch {
        command: command.clone(),
        protocol_command: [command, vec!["--resident-protocol-info".into()]].concat(),
        environment: env,
        cwd: root.clone(),
    };
    let metadata = json!({"profile":{"calibration_total_wall_seconds":12.0},"model_artifact_hash":identity.model_artifact_hash});
    let legacy_dir = root.join("legacy-worker");
    fs::create_dir(&legacy_dir).unwrap();
    launch
        .environment
        .insert("FIXTURE_LEGACY_PROBE".into(), "1".into());
    let mut legacy = SearchSession::new();
    let error = legacy
        .run(
            &identity,
            &launch,
            &[1, 0, 2, 3],
            &legacy_dir,
            Duration::from_secs(5),
            &metadata,
        )
        .unwrap_err();
    assert!(
        error.to_string().contains("loaded graph binding"),
        "{error}"
    );
    assert!(legacy.pids().is_empty());
    assert!(!legacy_dir.join("resident-session").exists());
    legacy.close().unwrap();
    launch.environment.remove("FIXTURE_LEGACY_PROBE");
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
        if index == 0 {
            assert_eq!(
                result.metadata["observed_parent_batch_by_rank_values"],
                json!([32, 64])
            );
        } else {
            assert_eq!(result.metadata["observed_parent_batch"], 32);
            assert!(result
                .metadata
                .get("observed_parent_batch_by_rank_values")
                .is_none());
        }
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
    let preparation = PreparationIdentity {
        graph_hash: Some(identity.graph_hash.clone()),
        geometry: [Some(4), Some(4), Some(2)],
        center: Some(identity.graph.center.clone()),
        generators: Some(identity.graph.generators.clone()),
        model_artifact_hash: identity.model_artifact_hash.clone(),
        artifact_sizes: Some(vec![17]),
        model_backend: identity.model_backend.clone(),
        devices: identity.devices.clone(),
        binding_scope: false,
        beam_width: identity.beam_width,
        max_steps: identity.max_steps,
        options_repr: "NativeOptions()".into(),
        cuda_visible_devices: None,
        cuda_device_order: None,
        runtime_build: Some(identity.runtime_build.clone()),
    };
    session.remember_preparation(&preparation).unwrap();
    assert!(session.begin_preparation(&preparation).unwrap());
    let mut changed_preparation = preparation.clone();
    changed_preparation.model_artifact_hash = "e".repeat(64);
    assert!(!session.begin_preparation(&changed_preparation).unwrap());
    assert!(session.pids().is_empty());

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
    identity.graph.center.swap(0, 1);
    let missing = root.join("stale-graph-must-not-be-created");
    let err = session
        .run(
            &identity,
            &launch,
            &identity.graph.center,
            &missing,
            Duration::from_secs(5),
            &metadata,
        )
        .unwrap_err();
    assert!(err.to_string().contains("graph hash"));
    assert!(session.pids().is_empty());
    assert!(!missing.exists());
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
 time.sleep(.3);print(json.dumps(dict(protocol='resident-search-v1',loaded_graph_binding=1)),flush=True);raise SystemExit(0)
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
        graph_hash: "47d5862a95983115a9c24781d00319694a06c0055a933113ffdee8c5ffb72309".into(),
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
