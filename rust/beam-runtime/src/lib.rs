//! Persistent session, rank process and history ownership.
#![forbid(unsafe_code)]
mod resident;
mod worker;
pub use resident::ResidentWorkers;
pub use worker::run_native_worker;

#[derive(Debug, thiserror::Error)]
pub enum Error {
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
