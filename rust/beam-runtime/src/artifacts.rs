//! Streaming content binding and atomic host artifacts. No model-sized buffers.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    if !file.metadata()?.is_file() {
        return Err(Error::Message(format!(
            "artifact is not a regular file: {}",
            path.display()
        )));
    }
    let mut buffer = vec![0u8; 64 << 10];
    let mut hash = Sha256::new();
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: Option<u64>,
}
impl Artifact {
    pub fn verify(&self) -> Result<()> {
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(Error::Message("invalid artifact SHA256".into()));
        }
        let metadata = fs::metadata(&self.path)?;
        if !metadata.is_file()
            || self.bytes.is_some_and(|size| size != metadata.len())
            || sha256_file(&self.path)? != self.sha256
        {
            return Err(Error::Message(format!(
                "artifact changed: {}",
                self.path.display()
            )));
        }
        Ok(())
    }
}
pub fn verify_all(artifacts: &[Artifact]) -> Result<()> {
    for artifact in artifacts {
        artifact.verify()?;
    }
    Ok(())
}
/// Linux rename replaces the old cache atomically. On unsupported filesystems an
/// error leaves the previous entry intact. Temporary files have unique owners.
pub fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .ok_or_else(|| Error::Message("artifact has no parent directory".into()))?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(
        ".beam-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = (|| {
        serde_json::to_writer(&mut file, value)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
