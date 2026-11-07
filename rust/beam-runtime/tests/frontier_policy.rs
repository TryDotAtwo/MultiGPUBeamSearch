use beam_core::calibration::Measurement;
use beam_runtime::{
    inference_admission::{measured_candidates, select_admitted},
    working_capacity, Error,
};
use serde_json::{json, Value};

fn calibration() -> Value {
    let mut rows = vec![];
    let mut telemetry = serde_json::Map::new();
    for (batch, seconds) in [(8, 0.2), (16, 0.1), (32, 0.09)] {
        for rank in 0..2 {
            rows.push(json!({"batch":batch,"parents":129,"device":rank,"seconds":vec![seconds;7],
                "correctness_passed":true,"numeric_error":0,"torch_reserved_peak_bytes":batch*1000}));
        }
        let sample = json!([{"index":0,"uuid":"GPU-a","temperature":50,"throttled":false},
            {"index":1,"uuid":"GPU-b","temperature":51,"throttled":false}]);
        telemetry.insert(
            batch.to_string(),
            json!({"samples":[sample.clone(),sample],
            "errors":[],"interval_seconds":5.,"throttled":false}),
        );
    }
    json!({"signature":{"world_size":2,"device_indices":[0,1],"gpu_uuids":["a","b"],"max_batch":32,
        "requested_beam_width":4097},"phase":"inference_verified","parent_batch":16,
        "reserve_bytes":16000+(512u64<<20),"records":rows,"gpu_telemetry":telemetry})
}

#[test]
fn fixture_state_budget_does_not_silently_cap_symbolic_maximum() {
    let center: Vec<u8> = (0..96).collect();
    let generator: Vec<u32> = (0..96).rev().collect();
    let artifact = json!({"path":"/fixture","sha256":"0".repeat(64),"bytes":null});
    let mut request: beam_runtime::frontier_job::Request = serde_json::from_value(json!({
        "beam_width":"max",
        "inference":{"options":{"calibration_frontier_max_states":8192},
            "graph":{"center":center,"generators":[generator]},"helper":artifact,
            "artifacts":[],"probe_manifest":{},"ensemble":false,
            "signature":{"world_size":1,"device_indices":[0],"gpu_uuids":["a"]},
            "beam_width":4096,"protocol":"oneshot","capacity_bootstrap":false,
            "run_dir":"/tmp/job","environment":{},"shared_profile":null},
        "execution":{"runner":artifact,"row_factor":1,"hardware":{"sm_count":1,"l2_bytes":0},
            "physical_bytes":112,"puzzle_id":0,"grow":true,"fixtures":[],"cwd":"/tmp",
            "gpu_memory_bytes":[1073741824]}
    }))
    .unwrap();
    assert!(request.capacity_ceiling().unwrap().upper_bound > 8192);
    request.execution.frontier_ceiling = Some(8192);
    assert_eq!(request.capacity_ceiling().unwrap().upper_bound, 8192);
}
fn plans(beam: u64) -> Vec<Value> {
    (0..2)
        .map(|rank| {
            json!({"GLOBAL_BEAM_WIDTH_EFFECTIVE":beam,"BEAM_WIDTH_ALIGNMENT":1024,
        "SHARD_COUNT":1,"B_MICRO":8,"WORLD_SIZE":2,"STREAM4_BATCH_ALIGNMENT":1024,
        "LOCAL_RANK":rank,"estimated_required_device_bytes":100,"gpu_budget_bytes":1000})
        })
        .collect()
}

#[test]
fn admission_changes_only_measured_batch_never_requested_width() {
    let data = calibration();
    let candidates = measured_candidates(&data, 32).unwrap();
    assert_eq!(
        candidates
            .iter()
            .map(|c| c.parent_batch)
            .collect::<Vec<_>>(),
        [16, 32, 8]
    );
    let mut calls = vec![];
    let (selected, rank_plans) = select_admitted(&data, 32, 4097, |candidate| {
        calls.push(candidate.parent_batch);
        if candidate.parent_batch >= 16 {
            Err(Error::CapacityRejected(
                "measured reserve does not fit".into(),
            ))
        } else {
            Ok(plans(4097))
        }
    })
    .unwrap();
    assert_eq!(calls, [16, 32, 8]);
    assert_eq!(selected["parent_batch"], 8);
    assert_eq!(selected["inference_admission"]["requested_beam"], 4097);
    assert_eq!(rank_plans[0]["GLOBAL_BEAM_WIDTH_EFFECTIVE"], 4097);
    let mut fatal_calls = 0;
    assert!(select_admitted(&data, 32, 4097, |_| {
        fatal_calls += 1;
        Err(Error::Message("rank died".into()))
    })
    .is_err());
    assert_eq!(fatal_calls, 1);
    assert!(select_admitted(&data, 32, 4097, |_| Ok(plans(4096))).is_err());
    assert!(
        select_admitted(&data, 32, 4097, |_| {
            let mut incomplete = plans(4097);
            incomplete.truncate(1);
            incomplete[0]["WORLD_SIZE"] = json!(1);
            Ok(incomplete)
        })
        .is_err(),
        "one valid rank cannot replace the requested two-rank cohort"
    );
    assert!(
        select_admitted(&data, 32, 4097, |_| {
            let mut unordered = plans(4097);
            unordered.swap(0, 1);
            Ok(unordered)
        })
        .is_err(),
        "the communicator rank order must match the requested cohort"
    );
}

#[test]
fn unsafe_or_incomplete_inference_cohorts_are_not_capacity_candidates() {
    let mut data = calibration();
    data["records"][4]["numeric_error"] = json!(1);
    data["gpu_telemetry"]["8"]["samples"][1][0]["throttled"] = json!(true);
    let candidates = measured_candidates(&data, 32).unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].parent_batch, 16);
    data["records"][3]["device"] = json!(0);
    assert!(measured_candidates(&data, 32).unwrap().is_empty());
}

#[test]
fn maximum_rebinds_exact_width_after_inference_reserve_changes() {
    use beam_runtime::frontier_job::refine_maximum;
    use std::cell::Cell;
    let calls = Cell::new(0);
    let admissions = Cell::new(0);
    let previous = calibration();
    let (capacity, inference) = refine_maximum(
        json!({"effective_beam":4097}),
        previous.clone(),
        32,
        |beam, _| {
            calls.set(calls.get() + 1);
            let mut value = previous.clone();
            value["signature"]["requested_beam_width"] = json!(beam);
            value["parent_batch"] = json!(8);
            value["reserve_bytes"] = json!(8000 + (512u64 << 20));
            Ok(value)
        },
        |chosen, _| {
            admissions.set(admissions.get() + 1);
            assert_eq!(chosen["parent_batch"], 8);
            Ok(json!({"effective_beam":4096}))
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 2);
    assert_eq!(admissions.get(), 1);
    assert_eq!(capacity["effective_beam"], 4096);
    assert_eq!(inference["signature"]["requested_beam_width"], 4096);
    assert_eq!(capacity["inference_refinement"][0]["readmitted"], true);
    assert_eq!(capacity["inference_refinement"][1]["readmitted"], false);
    assert!(refine_maximum(
        json!({"effective_beam":4097}),
        previous.clone(),
        32,
        |_, _| {
            let mut invalid = previous.clone();
            invalid["rejected"] = json!({"16":"GPU telemetry missing"});
            Ok(invalid)
        },
        |_, _| unreachable!()
    )
    .is_err());
    assert!(refine_maximum(
        json!({"effective_beam":4096}),
        previous.clone(),
        32,
        |_, _| Ok(previous.clone()),
        |_, _| unreachable!()
    )
    .is_err());
}

fn full(beam: u64, seconds: f64) -> Value {
    let rows: Vec<_> = (0..5)
        .map(|_| Measurement {
            profile: "baseline".into(),
            parents: beam,
            seconds_by_rank: vec![seconds * beam as f64; 2],
            correctness_passed: true,
            memory_admitted: true,
            throttled: false,
        })
        .collect();
    json!({"phase":"pipeline_measured","pipeline_verified":true,"measurement_scope":"full_requested_frontier",
        "requested_beam_effective":beam,"selection":"baseline","environment":{"BEAM_ENSEMBLE_INFERENCE_MICRO":"8"},
        "estimate":beam_core::calibration::estimate("baseline", &rows).unwrap(),"measurements":rows})
}
#[test]
fn working_width_requires_measured_same_profile_and_saturated_reference() {
    let mut widths = vec![];
    let result = working_capacity::choose(4096, full(4096, 2.), 8, 2, |width, _| {
        widths.push(width);
        Ok(full(width, 1.))
    })
    .unwrap();
    assert_eq!(widths, [2048, 1024]);
    assert_eq!(result.beam, 2048);
    assert!(result.efficiency_verified);
    assert_eq!(result.comparisons.len(), 2);
    assert!(working_capacity::choose(16, full(16, 1.), 8, 2, |_, _| unreachable!()).is_err());
    assert!(
        working_capacity::choose(4096, full(4096, 1.), 8, 2, |width, _| {
            let mut row = full(width, 1.);
            row["environment"]["BEAM_ENSEMBLE_INFERENCE_MICRO"] = json!("16");
            Ok(row)
        })
        .is_err()
    );
    let mut forged = full(4096, 1.);
    forged["estimate"]["median"] = json!(0.00001);
    assert!(working_capacity::require_full_step(4096, 2, &forged).is_err());
}
