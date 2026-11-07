//! Persistent JSON-line calibration services with owned process groups.
use crate::{Error, Result};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
const MAX_LINE: usize = 1 << 20;
type Row = std::result::Result<Value, String>;
fn until(duration: Duration) -> Result<Instant> {
    if duration.is_zero() {
        return Err(Error::Message("probe timed out".into()));
    }
    Instant::now()
        .checked_add(duration)
        .ok_or_else(|| Error::Message("probe deadline overflow".into()))
}
pub struct ProbeSession {
    child: Child,
    input: Option<ChildStdin>,
    rows: Option<Receiver<Row>>,
    reader: Option<JoinHandle<()>>,
    deadline: Instant,
    owner: u32,
    closed: bool,
    cancellation: Arc<AtomicBool>,
}
impl ProbeSession {
    pub fn launch(
        command: &[String],
        cwd: &Path,
        env: &BTreeMap<String, String>,
        log: &Path,
        timeout: Duration,
    ) -> Result<Self> {
        if command.is_empty() {
            return Err(Error::Message("empty probe command".into()));
        }
        let deadline = until(timeout)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(log)?;
        drop(file);
        let mut file = OpenOptions::new().append(true).open(log)?;
        let mut builder = Command::new(&command[0]);
        builder
            .args(&command[1..])
            .current_dir(cwd)
            .env_clear()
            .envs(env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(file.try_clone()?);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            builder.process_group(0);
        }
        let mut child = builder.spawn()?;
        let input = child.stdin.take();
        let output = child
            .stdout
            .take()
            .ok_or_else(|| Error::Message("probe stdout missing".into()))?;
        let (send, rows) = mpsc::sync_channel(64);
        let reader = thread::Builder::new()
            .name("beam-probe-log".into())
            .spawn(move || {
                let mut reader = BufReader::new(output);
                let mut line = Vec::new();
                loop {
                    line.clear();
                    match reader
                        .by_ref()
                        .take(MAX_LINE as u64 + 1)
                        .read_until(b'\n', &mut line)
                    {
                        Ok(0) => break,
                        Err(error) => {
                            let _ = send.send(Err(format!("probe output: {error}")));
                            break;
                        }
                        Ok(_) => {}
                    }
                    if let Err(error) = file.write_all(&line).and_then(|_| file.flush()) {
                        let _ = send.send(Err(format!("probe log: {error}")));
                        break;
                    }
                    if line.len() > MAX_LINE {
                        let _ = send.send(Err("probe response line exceeds 1 MiB".into()));
                        break;
                    }
                    if let Ok(value) = serde_json::from_slice::<Value>(&line) {
                        if value.is_object() && send.send(Ok(value)).is_err() {
                            break;
                        }
                    }
                }
            });
        let reader = match reader {
            Ok(reader) => reader,
            Err(error) => {
                let _ = beam_native_sys::kill_process_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        };
        Ok(Self {
            child,
            input,
            rows: Some(rows),
            reader: Some(reader),
            deadline,
            owner: std::process::id(),
            closed: false,
            cancellation: Arc::new(AtomicBool::new(false)),
        })
    }
    fn owner(&self) -> Result<()> {
        if self.owner != std::process::id() {
            Err(Error::Message(
                "probe cannot be shared across processes".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancellation.clone()
    }
    pub fn exit_code(&mut self) -> Result<Option<i32>> {
        self.owner()?;
        Ok(self.child.try_wait()?.map(|s| s.code().unwrap_or(-1)))
    }
    pub fn set_deadline(&mut self, timeout: Duration) -> Result<()> {
        self.owner()?;
        self.deadline = until(timeout)?;
        Ok(())
    }
    pub fn send(&mut self, request: &Value) -> Result<()> {
        self.owner()?;
        if self.cancellation.load(Ordering::Acquire) {
            return Err(Error::Message("probe cancelled".into()));
        }
        if self.closed || self.exit_code()?.is_some() {
            return Err(Error::Message("probe is no longer running".into()));
        }
        let mut bytes = serde_json::to_vec(request)?;
        if !request.is_object() || bytes.len() > MAX_LINE {
            return Err(Error::Message("invalid probe request".into()));
        }
        bytes.push(b'\n');
        let input = self
            .input
            .as_mut()
            .ok_or_else(|| Error::Message("probe input closed".into()))?;
        input.write_all(&bytes)?;
        input.flush()?;
        Ok(())
    }
    pub(crate) fn try_receive(&mut self) -> Result<Option<Value>> {
        self.owner()?;
        if self.cancellation.load(Ordering::Acquire) {
            return Err(Error::Message("probe cancelled".into()));
        }
        let rows = self
            .rows
            .as_ref()
            .ok_or_else(|| Error::Message("probe closed".into()))?;
        match rows.try_recv() {
            Ok(Ok(value)) => Ok(Some(value)),
            Ok(Err(error)) => Err(Error::Message(error)),
            Err(TryRecvError::Disconnected) => {
                Err(Error::Message("probe exited; inspect its log".into()))
            }
            Err(TryRecvError::Empty) => {
                if Instant::now() >= self.deadline {
                    return Err(Error::Message("probe timed out".into()));
                }
                Ok(None)
            }
        }
    }
    pub fn receive(&mut self, timeout: Duration) -> Result<Value> {
        let deadline = until(timeout)?.min(self.deadline);
        let result = (|| loop {
            if let Some(value) = self.try_receive()? {
                return Ok(value);
            }
            if Instant::now() >= deadline {
                return Err(Error::Message("probe timed out".into()));
            }
            thread::sleep(Duration::from_millis(5));
        })();
        if result.is_err() {
            let _ = self.close();
        }
        result
    }
    pub(crate) fn begin_close(&mut self) -> Result<()> {
        self.owner()?;
        self.closed = true;
        self.input.take();
        self.rows.take();
        Ok(())
    }
    pub(crate) fn finish_close(&mut self, deadline: Instant) -> Result<()> {
        self.owner()?;
        while self.child.try_wait()?.is_none() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        if self.child.try_wait()?.is_none()
            || self.reader.as_ref().is_some_and(|r| !r.is_finished())
        {
            beam_native_sys::kill_process_group(self.child.id())?;
            let _ = self.child.kill();
        }
        self.child.wait()?;
        if let Some(reader) = self.reader.take() {
            reader
                .join()
                .map_err(|_| Error::Message("probe log reader panicked".into()))?;
        }
        Ok(())
    }
    pub fn close(&mut self) -> Result<()> {
        self.begin_close()?;
        self.finish_close(Instant::now() + Duration::from_secs(2))
    }
    pub(crate) fn bind_cancellation(&mut self, cancellation: Arc<AtomicBool>) {
        self.cancellation = cancellation;
    }
    pub(crate) fn finish(&mut self, timeout: Duration) -> Result<()> {
        let result = (|| {
            let deadline = until(timeout)?.min(self.deadline);
            self.owner()?;
            self.input.take();
            loop {
                if self.cancellation.load(Ordering::Acquire) {
                    return Err(Error::Message("probe cancelled".into()));
                }
                if let Some(status) = self.child.try_wait()? {
                    return if status.success() {
                        Ok(())
                    } else {
                        Err(Error::Message(format!(
                            "probe exited with {}",
                            status.code().unwrap_or(-1)
                        )))
                    };
                }
                if Instant::now() >= deadline {
                    return Err(Error::Message("probe timed out".into()));
                }
                thread::sleep(Duration::from_millis(5));
            }
        })();
        let cleanup = self.close();
        result?;
        cleanup
    }
}
impl Drop for ProbeSession {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

pub struct ProbeCohort {
    sessions: Vec<ProbeSession>,
    cancellation: Arc<AtomicBool>,
}
impl ProbeCohort {
    pub fn bind_cancellation(&mut self, cancellation: Arc<AtomicBool>) {
        self.cancellation = cancellation;
        for session in &mut self.sessions {
            session.cancellation = self.cancellation.clone();
        }
    }
    pub fn launch(
        commands: &[Vec<String>],
        environments: &[BTreeMap<String, String>],
        cwd: &Path,
        directory: &Path,
        timeout: Duration,
    ) -> Result<Self> {
        if commands.is_empty() || commands.len() != environments.len() {
            return Err(Error::Message("invalid probe cohort".into()));
        }
        fs::create_dir(directory)?;
        let mut group = Self {
            sessions: Vec::new(),
            cancellation: Arc::new(AtomicBool::new(false)),
        };
        for (rank, (command, env)) in commands.iter().zip(environments).enumerate() {
            let mut session = ProbeSession::launch(
                command,
                cwd,
                env,
                &directory.join(format!("rank-{rank}.log")),
                timeout,
            )?;
            session.cancellation = group.cancellation.clone();
            group.sessions.push(session);
        }
        Ok(group)
    }
    pub fn pids(&self) -> Vec<u32> {
        self.sessions.iter().map(ProbeSession::pid).collect()
    }
    pub fn cancellation(&self) -> Arc<AtomicBool> {
        self.cancellation.clone()
    }
    pub fn set_deadline(&mut self, timeout: Duration) -> Result<()> {
        for session in &mut self.sessions {
            session.set_deadline(timeout)?;
        }
        Ok(())
    }
    pub fn exit_codes(&mut self) -> Result<Vec<Option<i32>>> {
        self.sessions
            .iter_mut()
            .map(ProbeSession::exit_code)
            .collect()
    }
    pub fn send(&mut self, requests: &[Value]) -> Result<()> {
        if requests.len() != self.sessions.len() {
            return Err(Error::Message("probe request omits ranks".into()));
        }
        for (session, request) in self.sessions.iter_mut().zip(requests) {
            if let Err(error) = session.send(request) {
                let _ = self.close();
                return Err(error);
            }
        }
        Ok(())
    }
    pub fn collect(&mut self, timeout: Duration) -> Result<Vec<Value>> {
        let deadline = until(timeout)?;
        let result = (|| {
            let mut responses = vec![None; self.sessions.len()];
            loop {
                for (session, response) in self.sessions.iter_mut().zip(&mut responses) {
                    if response.is_none() {
                        *response = session.try_receive()?;
                    }
                }
                if responses.iter().all(Option::is_some) {
                    return Ok(responses.into_iter().map(Option::unwrap).collect());
                }
                if Instant::now() >= deadline {
                    return Err(Error::Message("probe cohort timed out".into()));
                }
                thread::sleep(Duration::from_millis(5));
            }
        })();
        if result.is_err() {
            let _ = self.close();
        }
        result
    }
    /// Finish ordinary coarse native jobs, draining JSON diagnostics without
    /// blocking their stdout reader. Poll every rank so a late rank's failure
    /// cannot leave an earlier rank waiting inside NCCL until the deadline.
    pub fn finish(&mut self, timeout: Duration) -> Result<()> {
        let result = (|| {
            let deadline = until(timeout)?;
            for session in &mut self.sessions {
                session.owner()?;
                session.input.take();
            }
            loop {
                if self.cancellation.load(Ordering::Acquire) {
                    return Err(Error::Message("probe cohort cancelled".into()));
                }
                let mut complete = true;
                for (rank, session) in self.sessions.iter_mut().enumerate() {
                    if let Some(rows) = &session.rows {
                        loop {
                            match rows.try_recv() {
                                Ok(Err(error)) => return Err(Error::Message(error)),
                                Ok(Ok(_)) => {}
                                Err(_) => break,
                            }
                        }
                    }
                    match session.child.try_wait()? {
                        Some(status) if !status.success() => {
                            return Err(Error::Message(format!(
                                "rank {rank} exited with {}",
                                status.code().unwrap_or(-1)
                            )))
                        }
                        None => complete = false,
                        _ => {}
                    }
                    if Instant::now() >= session.deadline && !complete {
                        return Err(Error::Message("probe cohort timed out".into()));
                    }
                }
                if complete {
                    return Ok(());
                }
                if Instant::now() >= deadline {
                    return Err(Error::Message("probe cohort timed out".into()));
                }
                thread::sleep(Duration::from_millis(5));
            }
        })();
        let cleanup = self.close();
        result?;
        cleanup
    }
    pub fn close(&mut self) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut failure = None;
        for session in &mut self.sessions {
            if let Err(e) = session.begin_close() {
                failure = Some(e);
            }
        }
        for session in &mut self.sessions {
            if let Err(e) = session.finish_close(deadline) {
                failure = Some(e);
            }
        }
        failure.map_or(Ok(()), Err)
    }
}
impl Drop for ProbeCohort {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
