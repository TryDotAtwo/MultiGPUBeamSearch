//! Public configuration. Python only normalizes path-like objects to strings.
use crate::ContractError;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceBackend {
    Auto,
    Libtorch,
    Cutlass,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SolutionMode {
    First,
    Collect,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct NativeOptions {
    pub source_dir: Option<PathBuf>,
    pub cache_dir: PathBuf,
    pub cutlass_dir: Option<PathBuf>,
    pub runner_path: Option<PathBuf>,
    pub devices: Option<Vec<u32>>,
    pub num_gpus: Option<u32>,
    pub inference_backend: InferenceBackend,
    pub timeout_seconds: f64,
    pub build_timeout_seconds: f64,
    pub touch_bfs_radius: u32,
    pub touch_bfs_max_entries: u64,
    pub solution_mode: SolutionMode,
    pub collect_extra_depths: u32,
    pub max_collected_solutions: u32,
    pub build_jobs: u32,
    pub warn_on_fallback: bool,
    pub autotune: bool,
    pub calibration_seconds: f64,
    pub calibration_max_batch: u64,
    pub calibration_pipeline_seconds: f64,
    pub calibration_frontier_max_states: Option<u64>,
    pub report_calibration: bool,
    pub calibration_full_frontier: bool,
    pub calibration_discover_capacity: bool,
}
impl Default for NativeOptions {
    fn default() -> Self {
        Self {
            source_dir: None,
            cache_dir: PathBuf::from(".local/cache"),
            cutlass_dir: None,
            runner_path: None,
            devices: None,
            num_gpus: None,
            inference_backend: InferenceBackend::Auto,
            timeout_seconds: 3600.,
            build_timeout_seconds: 1800.,
            touch_bfs_radius: 0,
            touch_bfs_max_entries: 1_048_576,
            solution_mode: SolutionMode::First,
            collect_extra_depths: 1,
            max_collected_solutions: 2000,
            build_jobs: 2,
            warn_on_fallback: true,
            autotune: true,
            calibration_seconds: 90.,
            calibration_max_batch: 8192,
            calibration_pipeline_seconds: 600.,
            calibration_frontier_max_states: None,
            report_calibration: true,
            calibration_full_frontier: false,
            calibration_discover_capacity: false,
        }
    }
}
impl NativeOptions {
    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, value) in [
            ("timeout_seconds", self.timeout_seconds),
            ("build_timeout_seconds", self.build_timeout_seconds),
            ("calibration_seconds", self.calibration_seconds),
            (
                "calibration_pipeline_seconds",
                self.calibration_pipeline_seconds,
            ),
        ] {
            if !value.is_finite()
                || value <= 0.
                || std::time::Duration::try_from_secs_f64(value).is_err()
            {
                return Err(ContractError::Invalid(name));
            }
        }
        if self.num_gpus.is_some_and(|n| n == 0 || n > 65535)
            || (self.num_gpus.is_some() && self.devices.is_some())
        {
            return Err(ContractError::Invalid("num_gpus/devices"));
        }
        if self.devices.as_ref().is_some_and(|d| {
            d.is_empty() || d.len() > 65535 || d.iter().collect::<BTreeSet<_>>().len() != d.len()
        }) {
            return Err(ContractError::Invalid("devices"));
        }
        if self.touch_bfs_radius > 12 || self.touch_bfs_max_entries == 0 {
            return Err(ContractError::Invalid("touch BFS bounds"));
        }
        if !(1..=1_000_000).contains(&self.max_collected_solutions) {
            return Err(ContractError::Invalid("max_collected_solutions"));
        }
        if self.build_jobs == 0 {
            return Err(ContractError::Invalid("build_jobs"));
        }
        if self.cache_dir.as_os_str().is_empty() {
            return Err(ContractError::Invalid("cache_dir"));
        }
        if !(1..=65536).contains(&self.calibration_max_batch)
            || self.calibration_frontier_max_states == Some(0)
        {
            return Err(ContractError::Invalid("calibration bounds"));
        }
        if self.calibration_discover_capacity && !self.autotune {
            return Err(ContractError::Invalid(
                "capacity discovery requires autotune",
            ));
        }
        Ok(())
    }
}
