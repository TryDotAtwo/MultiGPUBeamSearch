//! Bounded full-step comparison. Component proxies only nominate challengers.
use crate::{
    artifacts,
    calibration::{self, Profile, ProfileEvaluator},
    component_tuning::{self, Selection},
    downstream_job::{Prepared, Request},
    pipeline::{Config, PipelineProbe},
    pipeline_cache, profile_cache, Error, Result,
};
use beam_core::calibration::{estimate, select};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs::File,
    io::Read,
    path::Path,
    sync::{atomic::AtomicBool, Arc},
    time::Instant,
};
#[derive(Serialize, Deserialize)]
struct Record {
    schema: u32,
    identity: Value,
    component_trials_hash: String,
    selection: calibration::Selection,
    full_tested: Vec<String>,
    full_grid_complete: bool,
}
fn shortlist(components: &Selection, prepared: &Prepared) -> Result<Vec<Profile>> {
    let mut valid = Vec::new();
    for trial in &components.tested {
        if trial.rejected.is_some() {
            continue;
        }
        let samples = component_tuning::validate_trial(&prepared.tuning, trial)?;
        if estimate(&trial.name, &samples).is_ok() {
            valid.push(trial);
        }
    }
    let base = valid
        .iter()
        .find(|r| r.name == "baseline")
        .ok_or_else(|| Error::Message("missing stable component baseline".into()))?;
    let mut selected = vec![Profile {
        name: "baseline".into(),
        environment: base.environment.clone(),
    }];
    valid.retain(|r| r.name != "baseline");
    valid.sort_by(|a, b| {
        a.estimate
            .as_ref()
            .unwrap()
            .service_work_seconds
            .total_cmp(&b.estimate.as_ref().unwrap().service_work_seconds)
            .then_with(|| a.name.cmp(&b.name))
    });
    if !valid.is_empty() {
        let row = valid.remove(0);
        selected.push(Profile {
            name: row.name.clone(),
            environment: row.environment.clone(),
        });
    }
    valid.sort_by_key(|r| {
        r.plans
            .iter()
            .filter_map(|p| p["estimated_required_device_bytes"].as_u64())
            .max()
            .unwrap_or(u64::MAX)
    });
    if let Some(row) = valid.first() {
        selected.push(Profile {
            name: row.name.clone(),
            environment: row.environment.clone(),
        });
    }
    Ok(selected)
}
fn valid_record(record: &Record, prepared: &Prepared, components: &Selection) -> Result<()> {
    if record.schema != 1
        || record.identity != prepared.identity
        || !record.full_grid_complete
        || record.component_trials_hash != profile_cache::cache_key(&json!(components.tested))?
        || record.selection.effective_beam != prepared.tuning.beam
        || record.selection.workload_parents != prepared.tuning.beam
        || record.selection.world_size != prepared.tuning.device_indices.len()
    {
        return Err(Error::Message("invalid full-step cache identity".into()));
    }
    let rows = &record.selection.measurements;
    let names: BTreeSet<_> = rows.iter().map(|m| m.profile.clone()).collect();
    let profiles = shortlist(components, prepared)?;
    let expected: BTreeSet<_> = profiles.iter().map(|r| r.name.clone()).collect();
    if names != expected
        || record.full_tested.iter().cloned().collect::<BTreeSet<_>>() != names
        || record.full_tested.len() != names.len()
        || rows.iter().any(|r| {
            r.parents != prepared.tuning.beam
                || r.seconds_by_rank.len() != prepared.tuning.device_indices.len()
        })
    {
        return Err(Error::Message("incomplete full-step cache cohort".into()));
    }
    let (winner, _) = select(rows, "baseline", 0.05)?;
    let selected = profiles
        .iter()
        .find(|r| r.name == winner.profile)
        .ok_or_else(|| Error::Message("unknown full-step winner".into()))?;
    if selected.environment != record.selection.environment
        || serde_json::to_value(winner)? != serde_json::to_value(&record.selection.estimate)?
    {
        return Err(Error::Message(
            "full-step cache does not reproduce selection".into(),
        ));
    }
    Ok(())
}
pub fn run(
    request: &Request,
    prepared: &Prepared,
    components: &Selection,
    directory: &Path,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
) -> Result<Value> {
    let timeout = deadline
        .checked_duration_since(Instant::now())
        .ok_or_else(|| Error::Message("full-step budget expired".into()))?;
    let mut owner = PipelineProbe::launch_cancellable(
        Config {
            runner: request.runner.path.clone(),
            environment: prepared.environment.clone(),
            beam: request.beam_width,
            world: prepared.tuning.device_indices.len() as u32,
            cwd: request.cwd.clone(),
            directory: directory.join("full-steps"),
            fixtures: request.fixtures.iter().map(|a| a.path.clone()).collect(),
            physical_bytes: request.physical_bytes,
            repeats: 6,
            puzzle_id: request.puzzle_id,
            grow: request.grow,
            gpu_uuids: prepared.tuning.gpu_uuids.clone(),
            device_indices: prepared.tuning.device_indices.clone(),
        },
        timeout,
        cancel,
    )?;
    let path = request
        .options
        .cache_dir
        .join("pipeline-profiles")
        .join(format!("{}-full.json", prepared.key));
    let read = (|| -> Result<Record> {
        let file = File::open(&path)?;
        if file.metadata()?.len() > 64 << 20 {
            return Err(Error::Message("oversize full-step cache".into()));
        }
        let mut bytes = Vec::new();
        file.take((64 << 20) + 1).read_to_end(&mut bytes)?;
        let record: Record = serde_json::from_slice(&bytes)?;
        valid_record(&record, prepared, components)?;
        Ok(record)
    })();
    let mut hit = None;
    if let Ok(record) = read {
        match owner.admit(&record.selection.environment) {
            Ok(plans) => {
                component_tuning::checked_plans(&prepared.tuning, &plans)?;
                let old = components
                    .tested
                    .iter()
                    .find(|r| r.name == record.selection.estimate.profile)
                    .ok_or_else(|| Error::Message("missing full-step selected geometry".into()))?;
                if pipeline_cache::same_geometry(&plans, &old.plans) {
                    hit = Some((record, plans));
                }
            }
            Err(Error::CapacityRejected(_) | Error::Contract(_)) => {}
            Err(error) => return Err(error),
        }
    }
    let (record, cache_hit, current) = if let Some((record, plans)) = hit {
        owner.close()?;
        (record, true, Some(plans))
    } else {
        let profiles = shortlist(components, prepared)?;
        let remaining = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64();
        let predicted = prepared.tuning.inference_seconds * request.beam_width as f64 * 6.;
        artifacts::atomic_json(
            &directory.join("verification-budget.json"),
            &json!({"parents":request.beam_width,
            "predicted_baseline_inference_seconds":predicted,"remaining_seconds":remaining,"fits":predicted<remaining}),
        )?;
        if !predicted.is_finite() || predicted >= remaining {
            return Err(Error::Message(
                "exact full-frontier baseline exceeds calibration time budget; width unchanged"
                    .into(),
            ));
        }
        let result = owner.tune(&profiles, prepared.tuning.micro);
        let cleanup = owner.close();
        let selection = result?;
        cleanup?;
        let tested: BTreeSet<_> = selection
            .measurements
            .iter()
            .map(|m| m.profile.clone())
            .collect();
        let complete =
            tested.len() == profiles.len() && profiles.iter().all(|p| tested.contains(&p.name));
        (
            Record {
                schema: 1,
                identity: prepared.identity.clone(),
                component_trials_hash: profile_cache::cache_key(&json!(components.tested))?,
                selection,
                full_tested: tested.into_iter().collect(),
                full_grid_complete: complete,
            },
            false,
            None,
        )
    };
    // Verify artifacts before any persistent full-step cache becomes visible.
    request.runner.verify()?;
    artifacts::verify_all(&request.artifacts)?;
    artifacts::verify_all(&request.fixtures)?;
    if record.full_grid_complete && !cache_hit {
        valid_record(&record, prepared, components)?;
        artifacts::atomic_json(&path, &record)?;
    }
    let mut data = serde_json::to_value(components)?;
    let winner = &record.selection.estimate;
    for (key, value) in [
        ("phase", json!("pipeline_measured")),
        ("pipeline_verified", json!(true)),
        ("cache_hit", json!(cache_hit)),
        ("selection", json!(winner.profile)),
        ("estimate", json!(winner)),
        ("environment", json!(record.selection.environment)),
        ("measurement_scope", json!("full_requested_frontier")),
        ("full_grid_complete", json!(record.full_grid_complete)),
        ("full_tested", json!(record.full_tested)),
        (
            "full_comparison_measurements",
            json!(record.selection.measurements),
        ),
        (
            "full_comparison_rejections",
            json!(record.selection.rejected),
        ),
        (
            "selection_objective",
            json!("statistically separated full-step time in the measured bounded shortlist"),
        ),
        (
            "full_selection_policy",
            json!("at_least_5_percent_separated_full_step_v1"),
        ),
        ("current_rank_plans", json!(current)),
    ] {
        data[key] = value;
    }
    data["measurements"] = json!(record
        .selection
        .measurements
        .iter()
        .filter(|r| r.profile == winner.profile)
        .collect::<Vec<_>>());
    Ok(data)
}
