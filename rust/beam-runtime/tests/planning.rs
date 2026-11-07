use beam_runtime::planning::PlanningSession;
use std::{collections::BTreeMap, fs, time::Duration};

// Protocol adversary: rank 1 rejects preparation while rank 0 is prepared.
// A run request before cohort-wide preparation is a fatal error in this fixture.
#[test]
fn planning_preserves_exact_width_freezes_inference_and_cancels_all_preparations() {
    let dir = std::env::temp_dir().join(format!("beam-plan-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let worker = dir.join("worker.py");
    fs::write(&worker, r#"import sys,os,json
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);prepared=False
print(json.dumps(dict(ready=True,rank=rank,component_protocol='persistent-exact-capacity-v1')),flush=True)
for line in sys.stdin:
 r=json.loads(line)
 if r.get('cancel_component'):
  prepared=False;answer=dict(cancelled=True,rank=rank)
 elif r.get('run_component'):
  assert prepared;prepared=False;answer=dict(rank=rank,component_probe=True)
 elif r.get('prepare_component'):
  prepared=r['beam']!=13 or rank!=1;answer=dict(prepared=prepared,rank=rank)
 elif r['beam']==99:answer=dict(capacity_rejected=True,reason='required=99 available=64')
 else:
  p=dict(GLOBAL_BEAM_WIDTH_EFFECTIVE=r['beam']+(r['beam']==17),BEAM_WIDTH_ALIGNMENT=32,
    WORLD_SIZE=world,LOCAL_RANK=rank,B_MICRO=64,SHARD_COUNT=1,STREAM4_BATCH_ALIGNMENT=32,
    estimated_required_device_bytes=512,gpu_budget_bytes=1024)
  answer=dict(admitted=True,plan=p)
 print(json.dumps(answer),flush=True)
"#).unwrap();
    let mut env: BTreeMap<String, String> = std::env::vars().collect();
    env.insert("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), "64".into());
    let command = vec!["python".into(), worker.to_string_lossy().into_owned()];
    let mut group = PlanningSession::launch(
        &command,
        &env,
        2,
        &dir,
        &dir.join("cohort"),
        Duration::from_secs(10),
    )
    .unwrap();
    let pids = group.pids();
    assert!(group.supports_components());
    assert_eq!(group.admit(5, &BTreeMap::new()).unwrap().len(), 2);
    assert!(group
        .admit(17, &BTreeMap::new())
        .unwrap_err()
        .to_string()
        .contains("exact"));
    assert!(group
        .admit(99, &BTreeMap::new())
        .unwrap_err()
        .to_string()
        .contains("required=99 available=64"));
    assert!(group
        .admit(
            5,
            &BTreeMap::from([("BEAM_ENSEMBLE_INFERENCE_MICRO".into(), "32".into())])
        )
        .is_err());
    assert!(group
        .admit(5, &BTreeMap::from([("BEAM_WIDTH".into(), "3".into())]))
        .is_err());
    assert!(group.measure_components(13, &BTreeMap::new()).is_err());
    assert_eq!(
        group.measure_components(5, &BTreeMap::new()).unwrap().len(),
        2
    );
    assert_eq!(group.pids(), pids);
    group.close().unwrap();
    assert!(group.exit_codes().unwrap().iter().all(Option::is_some));
    for rank in 0..2 {
        let log = fs::read_to_string(dir.join(format!("cohort/rank-{rank}.log"))).unwrap();
        assert!(log.contains("\"cancelled\": true"));
    }
    fs::remove_dir_all(dir).unwrap();
}
