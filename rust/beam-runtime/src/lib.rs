//! Persistent session, rank process and history ownership.
#![forbid(unsafe_code)]
pub mod artifacts;
pub mod calibration;
pub mod component_probe;
pub mod component_tuning;
mod downstream_full;
pub mod downstream_job;
pub mod frontier_job;
mod frontier_native;
pub mod history;
pub mod inference;
pub mod inference_admission;
pub mod inference_job;
pub mod inference_tuning;
pub mod maximum_beam;
pub mod pipeline;
pub mod pipeline_cache;
pub mod planning;
pub mod probes;
pub mod profile_cache;
mod resident;
pub mod search_session;
pub mod telemetry;
mod worker;
pub mod working_capacity;
pub use resident::ResidentWorkers;
pub use worker::run_native_worker;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("capacity rejected: {0}")]
    CapacityRejected(String),
    #[error("{0}")]
    Message(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Native(#[from] beam_native_sys::Error),
    #[error(transparent)]
    Contract(#[from] beam_core::ContractError),
}
pub type Result<T> = std::result::Result<T, Error>;

#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct Request {
    pub sequence: u64,
    pub start: Vec<u8>,
    pub run_dir: std::path::PathBuf,
}
#[derive(serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    pub sequence: u64,
    pub rank: u32,
}
pub(crate) fn atomic_json(path: &std::path::Path, value: &impl serde::Serialize) -> Result<()> {
    use std::io::Write;
    let temporary = path.with_extension("tmp");
    let mut file = std::fs::File::create(&temporary)?;
    serde_json::to_writer(&mut file, value)?;
    file.flush()?;
    drop(file);
    std::fs::rename(temporary, path)?;
    Ok(())
}
