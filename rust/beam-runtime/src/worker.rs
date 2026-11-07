use crate::{atomic_json, Error, Receipt, Request, Result};
use beam_core::{Domain, Rank, SearchConfig};
use beam_native_sys::NativeLibrary;
use std::{env, fs, path::Path, thread, time::Duration};

fn number(name: &str, default: u32) -> Result<u32> {
    env::var(name).map_or(Ok(default), |s| {
        s.parse()
            .map_err(|_| Error::Message(format!("invalid {name}")))
    })
}
/// Coarse bridge used by a single isolated rank process. All mailbox polling
/// and process supervision live in Rust; native C++ owns the GPU search loop.
pub fn run_native_worker(library: &NativeLibrary, args: &[String], root: &Path) -> Result<()> {
    if args.len() != 4 && args.len() != 6 {
        return Err(Error::Message(
            "worker requires puzzle/depth/beam [world/rank]".into(),
        ));
    }
    let positional = |index: usize, default: u32| -> Result<u32> {
        args.get(index).map_or(Ok(default), |s| {
            s.parse()
                .map_err(|_| Error::Message("invalid rank argument".into()))
        })
    };
    let world = number("WORLD_SIZE", positional(4, 1)?)?;
    let rank = number("RANK", positional(5, 0)?)?;
    let layout = library.layout()?;
    let config = SearchConfig {
        domain: Domain::permutation(
            layout.state_len as u64,
            layout.move_count,
            layout.state_alignment,
        )?,
        beam_width: args[3]
            .parse()
            .map_err(|_| Error::Message("invalid beam width".into()))?,
        max_depth: args[2]
            .parse()
            .map_err(|_| Error::Message("invalid depth".into()))?,
        rank: Rank::new(
            world,
            rank,
            number("LOCAL_WORLD_SIZE", world)?,
            number("LOCAL_RANK", rank)?,
        )?,
    };
    config.validate()?;
    if config.domain.state_bytes()? != layout.storage_len as u64
        || (
            layout.candidate_bytes,
            layout.candidate_alignment,
            layout.hash_bytes,
            layout.hash_alignment,
        ) != (32, 32, 16, 16)
    {
        return Err(Error::Message(
            "native binary layout does not satisfy Rust contracts".into(),
        ));
    }
    let root = root.canonicalize()?;
    let owner = beam_native_sys::parent_process_id();
    let stopped = || root.join("stop").exists() || beam_native_sys::parent_process_id() != owner;
    let mut context = library.context()?;
    context.prepare(args)?;
    while !context.wait(Duration::from_millis(20))? {
        if stopped() {
            context.cancel()?;
            return Ok(());
        }
    }
    if context.status()?.phase != 2 {
        return Err(Error::Message(
            "native engine exited before session preparation".into(),
        ));
    }
    for sequence in 0..u64::MAX {
        let path = root.join(format!("request-{sequence}.json"));
        while !path.exists() {
            if stopped() {
                context.cancel()?;
                return Ok(());
            }
            thread::sleep(Duration::from_millis(10));
        }
        if stopped() {
            context.cancel()?;
            return Ok(());
        }
        if fs::metadata(&path)?.len() > 65536 {
            return Err(Error::Message("resident request too large".into()));
        }
        let request: Request = serde_json::from_slice(&fs::read(path)?)?;
        if request.sequence != sequence || request.start.len() != layout.state_len as usize {
            return Err(Error::Message(
                "resident request sequence/state length mismatch".into(),
            ));
        }
        context.submit(sequence, &request.start, &request.run_dir)?;
        while !context.wait(Duration::from_millis(20))? {
            if stopped() {
                context.cancel()?;
                return Ok(());
            }
        }
        if context.status()?.completed_jobs != sequence + 1 {
            return Err(Error::Message(
                "native engine did not complete the submitted job".into(),
            ));
        }
        atomic_json(
            &root.join(format!("done-{sequence}-{rank}.json")),
            &Receipt { sequence, rank },
        )?;
    }
    Err(Error::Message("request sequence overflow".into()))
}
