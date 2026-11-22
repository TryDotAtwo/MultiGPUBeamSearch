use crate::{atomic_json, Error, Receipt, Request, Result};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// Owns only processes it launched. No shell, global process search, serialized
/// device pointers, or access to another session's process group.
pub struct ResidentWorkers {
    directory: PathBuf,
    children: Vec<Child>,
    sequence: u64,
    closed: bool,
    owner_pid: u32,
    cancel: Arc<AtomicBool>,
    parent_cancel: Option<Arc<AtomicBool>>,
}
impl ResidentWorkers {
    pub fn launch(
        command: &[String],
        world: u32,
        directory: &Path,
        cwd: &Path,
        env: &BTreeMap<String, String>,
    ) -> Result<Self> {
        if command.is_empty() || world == 0 || world > 65535 {
            return Err(Error::Message("invalid resident command/world".into()));
        }
        let cwd = cwd.canonicalize()?;
        let directory = if directory.is_absolute() {
            directory.to_owned()
        } else {
            std::env::current_dir()?.join(directory)
        };
        if let Some(parent) = directory.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(&directory)?;
        let mut group = Self {
            directory,
            children: Vec::new(),
            sequence: 0,
            closed: false,
            owner_pid: std::process::id(),
            cancel: Arc::new(AtomicBool::new(false)),
            parent_cancel: None,
        };
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::Message(e.to_string()))?
            .as_nanos();
        let rendezvous = format!("resident-{}-{nonce}", group.owner_pid);
        for rank in 0..world {
            let log = File::create(group.directory.join(format!("bootstrap-{rank}.log")))?;
            let mut child = Command::new(&command[0]);
            child
                .args(&command[1..])
                .current_dir(&cwd)
                .env_clear()
                .envs(env)
                .env("BEAM_SESSION_DIR", &group.directory)
                .env("WORLD_SIZE", world.to_string())
                .env("RANK", rank.to_string())
                .env("LOCAL_RANK", rank.to_string())
                .env("LOCAL_WORLD_SIZE", world.to_string())
                .env("BEAM_NCCL_RUN_ID", &rendezvous)
                .stdin(Stdio::null())
                .stdout(log.try_clone()?)
                .stderr(log);
            #[cfg(unix)]
            {
                use std::os::unix::process::CommandExt;
                child.process_group(0);
            }
            group.children.push(child.spawn()?);
        }
        Ok(group)
    }
    fn check_owner(&self) -> Result<()> {
        if self.owner_pid != std::process::id() {
            Err(Error::Message(
                "native session cannot be shared across processes".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub fn pids(&self) -> Vec<u32> {
        self.children.iter().map(Child::id).collect()
    }
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
    pub fn is_closed(&self) -> bool {
        self.closed
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancel.clone()
    }
    pub(crate) fn bind_parent_cancellation(&mut self, cancel: Arc<AtomicBool>) {
        self.parent_cancel = Some(cancel);
    }
    pub fn exit_codes(&mut self) -> Result<Vec<Option<i32>>> {
        self.check_owner()?;
        self.children
            .iter_mut()
            .map(|p| Ok(p.try_wait()?.map(|s| s.code().unwrap_or(-1))))
            .collect()
    }
    pub fn submit(&mut self, start: &[u8], run_dir: &Path, timeout: Duration) -> Result<Duration> {
        self.check_owner()?;
        if self.closed {
            return Err(Error::Message("resident worker group is closed".into()));
        }
        let result = self.submit_inner(start, run_dir, timeout);
        if result.is_err() {
            let _ = self.close();
        }
        result
    }
    fn submit_inner(
        &mut self,
        start: &[u8],
        run_dir: &Path,
        timeout: Duration,
    ) -> Result<Duration> {
        if timeout.is_zero() {
            return Err(Error::Message("search timeout must be positive".into()));
        }
        let directory = run_dir.canonicalize()?;
        if !directory.is_dir() {
            return Err(Error::Message("output directory must exist".into()));
        }
        let began = Instant::now();
        let request = Request {
            sequence: self.sequence,
            start: start.to_vec(),
            run_dir: directory.clone(),
        };
        let path = self
            .directory
            .join(format!("request-{}.json", self.sequence));
        if serde_json::to_vec(&request)?.len() > 65536 {
            return Err(Error::Message("resident request too large".into()));
        }
        atomic_json(&path, &request)?;
        loop {
            if self.cancel.load(Ordering::Acquire)
                || self
                    .parent_cancel
                    .as_ref()
                    .is_some_and(|v| v.load(Ordering::Acquire))
            {
                return Err(Error::Message("resident search cancelled".into()));
            }
            let codes = self.exit_codes()?;
            if codes.iter().any(Option::is_some) {
                return Err(Error::Message(format!(
                    "resident worker exited: {codes:?}; logs={}",
                    self.directory.display()
                )));
            }
            let mut complete = true;
            for rank in 0..self.children.len() as u32 {
                let path = self
                    .directory
                    .join(format!("done-{}-{rank}.json", self.sequence));
                if !path.is_file() {
                    complete = false;
                    break;
                }
                if fs::metadata(&path)?.len() > 1024 {
                    return Err(Error::Message("invalid resident completion receipt".into()));
                }
                let actual: Receipt = serde_json::from_slice(&fs::read(path)?)?;
                if actual
                    != (Receipt {
                        sequence: self.sequence,
                        rank,
                    })
                {
                    return Err(Error::Message("invalid resident completion receipt".into()));
                }
            }
            if complete {
                break;
            }
            if began.elapsed() >= timeout {
                return Err(Error::Message(format!(
                    "resident search timed out; logs={}",
                    self.directory.display()
                )));
            }
            thread::sleep(Duration::from_millis(10));
        }
        let mut output = File::create(directory.join("native.log"))?;
        for rank in 0..self.children.len() {
            writeln!(output, "[rank {rank}]")?;
            // Stream the log; do not make another full-sized in-memory copy.
            let mut input = File::open(directory.join(format!("resident-rank-{rank}.log")))?;
            std::io::copy(&mut input, &mut output)?;
            writeln!(output)?;
        }
        output.flush()?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| Error::Message("request sequence overflow".into()))?;
        Ok(began.elapsed())
    }
    pub fn close(&mut self) -> Result<()> {
        self.check_owner()?;
        if self.closed {
            return Ok(());
        }
        self.closed = true;
        self.cancel.store(true, Ordering::Release);
        let mut errors = Vec::new();
        if let Err(e) = File::create(self.directory.join("stop")) {
            errors.push(e.to_string());
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let mut running = false;
            for child in &mut self.children {
                match child.try_wait() {
                    Ok(None) => running = true,
                    Ok(Some(_)) => {}
                    Err(e) => errors.push(e.to_string()),
                }
            }
            if !running || Instant::now() >= deadline {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        for child in &mut self.children {
            match child.try_wait() {
                Ok(Some(_)) => continue,
                Ok(None) => {}
                Err(e) => errors.push(e.to_string()),
            }
            if let Err(e) = beam_native_sys::kill_process_group(child.id()) {
                errors.push(e.to_string());
            }
            // Also supports platforms without Unix process groups.
            let _ = child.kill();
            if let Err(e) = child.wait() {
                errors.push(e.to_string());
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(Error::Message(format!(
                "failed to reap resident workers: {}",
                errors.join("; ")
            )))
        }
    }
}
impl Drop for ResidentWorkers {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
