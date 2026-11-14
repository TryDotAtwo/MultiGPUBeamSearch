//! Session-owned snapshots of an already validated MLP artifact.
//!
//! The caller validates the manifest, tensor shapes and finite values first.
//! This module binds those checked bytes to the copy using the same artifact
//! digest and bounded host scratch; it does not interpret device pointers.
use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, AtomicU64, Ordering},
};

#[derive(Clone, Debug)]
pub struct MlpSnapshotInput {
    pub source_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub run_dir: PathBuf,
    pub graph_hash: String,
    pub artifact_hash: String,
    /// Exact lexicographically ordered runtime tensor filenames.
    pub tensors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MlpSnapshot {
    pub path: PathBuf,
    pub reused: bool,
    pub copied_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Key {
    cache_dir: PathBuf,
    graph_hash: String,
    artifact_hash: String,
    tensors: Vec<String>,
}

#[derive(Default)]
pub struct MlpSnapshotStore {
    entry: Option<(Key, PathBuf)>,
}

impl MlpSnapshotStore {
    pub fn clear(&mut self) {
        self.entry = None;
    }

    pub fn prepare(&mut self, input: &MlpSnapshotInput, force_new: bool) -> Result<MlpSnapshot> {
        self.prepare_with_cancel(input, force_new, &AtomicBool::new(false))
    }

    pub fn prepare_with_cancel(
        &mut self,
        input: &MlpSnapshotInput,
        force_new: bool,
        cancel: &AtomicBool,
    ) -> Result<MlpSnapshot> {
        self.prepare_inner(input, force_new, cancel, false)
    }

    /// Defer validating a reused snapshot to the caller's launch boundary.
    /// The session must verify the returned snapshot immediately before use.
    pub fn prepare_for_session_with_cancel(
        &mut self,
        input: &MlpSnapshotInput,
        force_new: bool,
        cancel: &AtomicBool,
    ) -> Result<MlpSnapshot> {
        self.prepare_inner(input, force_new, cancel, true)
    }

    fn prepare_inner(
        &mut self,
        input: &MlpSnapshotInput,
        force_new: bool,
        cancel: &AtomicBool,
        defer_reused_snapshot_verification: bool,
    ) -> Result<MlpSnapshot> {
        validate(input)?;
        check_cancelled(cancel)?;
        let key = Key {
            // The public adapter supplies a resolved absolute path, even when
            // that cache directory has not been created yet.
            cache_dir: if input.cache_dir.is_absolute() {
                input.cache_dir.clone()
            } else {
                std::env::current_dir()?.join(&input.cache_dir)
            },
            graph_hash: input.graph_hash.clone(),
            artifact_hash: input.artifact_hash.clone(),
            tensors: input.tensors.clone(),
        };
        if !force_new {
            if let Some((old_key, snapshot)) = &self.entry {
                if old_key == &key {
                    verify(&input.source_dir, input, cancel)?;
                    if !defer_reused_snapshot_verification {
                        verify(snapshot, input, cancel)?;
                    }
                    return Ok(MlpSnapshot {
                        path: snapshot.clone(),
                        reused: true,
                        copied_bytes: 0,
                    });
                }
            }
        }

        let target = input.run_dir.join("weights");
        if input.source_dir == target {
            verify(&input.source_dir, input, cancel)?;
            self.entry = Some((key, target.clone()));
            return Ok(MlpSnapshot {
                path: target,
                reused: false,
                copied_bytes: 0,
            });
        }
        fs::create_dir_all(&input.run_dir)?;
        match target.symlink_metadata() {
            Ok(_) => {
                return Err(Error::Message(format!(
                    "private native weight snapshot already exists at {}",
                    target.display()
                )));
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let staging = Staging::create(&input.run_dir)?;
        let copied_bytes = hash_and_copy(&input.source_dir, Some(&staging.path), input, cancel)?;
        // Check the bytes that actually reached the private directory before
        // publishing it, not merely the bytes that were read from the source.
        verify(&staging.path, input, cancel)?;
        // Cancellation is cooperative. After this point publication may
        // complete even if a cancellation request arrives concurrently.
        check_cancelled(cancel)?;
        publish(&staging.path, &target)?;
        staging.disarm();
        self.entry = Some((key, target.clone()));
        Ok(MlpSnapshot {
            path: target,
            reused: false,
            copied_bytes,
        })
    }
}

#[cfg(target_os = "linux")]
fn publish(staging: &Path, target: &Path) -> Result<()> {
    use rustix::fs::{renameat_with, RenameFlags, CWD};
    renameat_with(CWD, staging, CWD, target, RenameFlags::NOREPLACE)
        .map_err(|error| Error::Message(format!("cannot publish MLP snapshot: {error}")))
}

#[cfg(not(target_os = "linux"))]
fn publish(staging: &Path, target: &Path) -> Result<()> {
    // Preserve the existing platform's rename semantics outside Linux.
    fs::rename(staging, target)?;
    Ok(())
}

fn validate(input: &MlpSnapshotInput) -> Result<()> {
    fn valid_hash(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    }
    if !valid_hash(&input.graph_hash) || !valid_hash(&input.artifact_hash) {
        return Err(Error::Message(
            "invalid MLP graph or artifact SHA256".into(),
        ));
    }
    if input.tensors.is_empty() {
        return Err(Error::Message("MLP tensor list is empty".into()));
    }
    let mut suffix = None;
    for (index, name) in input.tensors.iter().enumerate() {
        let (stem, extension) = name.rsplit_once('.').unwrap_or(("", ""));
        if stem.is_empty()
            || !stem.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
            || !matches!(extension, "fp16" | "bf16")
            || suffix.is_some_and(|old| old != extension)
            || (index > 0 && input.tensors[index - 1] >= *name)
        {
            return Err(Error::Message(format!(
                "invalid or unordered MLP tensor name: {name}"
            )));
        }
        suffix = Some(extension);
    }
    Ok(())
}

fn check_cancelled(cancel: &AtomicBool) -> Result<()> {
    if cancel.load(Ordering::Acquire) {
        return Err(Error::Message("MLP snapshot cancelled".into()));
    }
    Ok(())
}

fn verify(path: &Path, input: &MlpSnapshotInput, cancel: &AtomicBool) -> Result<()> {
    hash_and_copy(path, None, input, cancel).map(|_| ())
}

fn hash_and_copy(
    source: &Path,
    destination: Option<&Path>,
    input: &MlpSnapshotInput,
    cancel: &AtomicBool,
) -> Result<u64> {
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut scratch = vec![0u8; 64 * 1024];
    for (index, name) in std::iter::once("manifest.json")
        .chain(input.tensors.iter().map(String::as_str))
        .enumerate()
    {
        check_cancelled(cancel)?;
        if index == 1 {
            hash.update(input.graph_hash.as_bytes());
        }
        if index > 0 {
            hash.update(name.as_bytes());
        }
        let path = source.join(name);
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // A FIFO with no writer must not block before its type is checked.
            options.custom_flags(libc::O_NONBLOCK);
        }
        let mut from = options.open(&path)?;
        if !from.metadata()?.is_file() {
            return Err(Error::Message(format!(
                "MLP input is not a regular file: {}",
                path.display()
            )));
        }
        let mut to = match destination {
            Some(dir) => Some(
                OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(dir.join(name))?,
            ),
            None => None,
        };
        loop {
            check_cancelled(cancel)?;
            let n = from.read(&mut scratch)?;
            if n == 0 {
                break;
            }
            hash.update(&scratch[..n]);
            if let Some(output) = to.as_mut() {
                output.write_all(&scratch[..n])?;
            }
            total = total
                .checked_add(n as u64)
                .ok_or_else(|| Error::Message("MLP snapshot byte count overflow".into()))?;
        }
        if let Some(output) = to.as_mut() {
            output.flush()?;
        }
    }
    let actual = format!("{:x}", hash.finalize());
    if actual != input.artifact_hash {
        return Err(Error::Message(format!(
            "MLP artifact hash mismatch: expected={} actual={actual}",
            input.artifact_hash
        )));
    }
    Ok(total)
}

struct Staging {
    path: PathBuf,
    armed: bool,
}
impl Staging {
    fn create(parent: &Path) -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        for _ in 0..128 {
            let path = parent.join(format!(
                ".weights-snapshot-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path, armed: true }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err(Error::Message(
            "could not allocate a unique MLP snapshot staging directory".into(),
        ))
    }
    fn disarm(mut self) {
        self.armed = false;
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        if self.armed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn publish_does_not_replace_an_existing_empty_directory() {
        let parent =
            std::env::temp_dir().join(format!("beam-snapshot-publish-{}", std::process::id()));
        fs::create_dir_all(&parent).unwrap();
        let staging = parent.join("staging");
        let target = parent.join("target");
        fs::create_dir(&staging).unwrap();
        fs::create_dir(&target).unwrap();
        fs::write(staging.join("owned"), b"keep").unwrap();
        assert!(publish(&staging, &target).is_err());
        assert_eq!(fs::read(staging.join("owned")).unwrap(), b"keep");
        assert_eq!(fs::read_dir(&target).unwrap().count(), 0);
        fs::remove_dir_all(parent).unwrap();
    }
}
