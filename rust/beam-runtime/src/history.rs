//! Host history owns its files, RAM, reservations and parent remapping in Rust.
//! Native callers lend a ready pinned batch only until `write` returns.
use crate::{Error, Result};
use beam_core::{
    history::{self, Entry, Location, Plan, Stats},
    require_capacity, CandidateMeta, ContractError,
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::PathBuf,
};

#[derive(Default)]
pub struct HistoryService(Mutex<Option<HistoryStore>>);
impl HistoryService {
    fn with<T>(
        &self,
        operation: impl FnOnce(&mut HistoryStore) -> Result<T>,
    ) -> std::result::Result<T, String> {
        let mut owner = self.0.lock().map_err(|e| e.to_string())?;
        let store = owner
            .as_mut()
            .ok_or_else(|| "history not initialized".to_string())?;
        operation(store).map_err(|e| e.to_string())
    }
}
impl beam_native_sys::history::HistorySink for HistoryService {
    fn initialize(&self, json: &[u8]) -> std::result::Result<(), String> {
        let config: HistoryConfig = serde_json::from_slice(json).map_err(|e| e.to_string())?;
        let mut owner = self.0.lock().map_err(|e| e.to_string())?;
        if owner.is_some() {
            return Err("history already initialized".into());
        }
        *owner = Some(HistoryStore::new(config).map_err(|e| e.to_string())?);
        Ok(())
    }
    fn write(&self, depth: u32, batch: &[CandidateMeta]) -> std::result::Result<(), String> {
        self.with(|store| store.write(depth, batch))
    }
    fn read(&self, depth: u32, index: u64) -> std::result::Result<Entry, String> {
        self.with(|store| store.read(depth, index))
    }
    fn stats(&self) -> std::result::Result<Stats, String> {
        self.with(|store| Ok(store.stats()))
    }
    fn close(&self) -> std::result::Result<(), String> {
        self.0.lock().map_err(|e| e.to_string())?.take();
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoryMode {
    Ram,
    Disk,
    StaticHybrid,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryConfig {
    pub mode: HistoryMode,
    pub capacity: u32,
    pub slots: u32,
    pub workers: u32,
    pub max_depth: u32,
    pub moves: u32,
    pub ram_budget_bytes: u64,
    pub disk_budget_bytes: u64,
    pub directory: PathBuf,
    pub disk_path: Option<PathBuf>,
    pub prune: bool,
}
enum Storage {
    Ram(Vec<Entry>),
    Disk(PathBuf),
    ArenaRam(u64),
    ArenaDisk(u64),
}
struct Depth {
    count: u64,
    storage: Storage,
}
pub struct HistoryStore {
    config: HistoryConfig,
    stats: Stats,
    plan: Option<Plan>,
    depths: Vec<Option<Depth>>,
    ram: Vec<Entry>,
    disk: Option<File>,
    disk_failed: bool,
    staging: Vec<u8>,
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or(ContractError::Overflow("history bytes").into())
}
fn allocate<T: Default + Clone>(count: u64) -> Result<Vec<T>> {
    let count =
        usize::try_from(count).map_err(|_| ContractError::Overflow("history allocation"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|e| Error::Message(format!("history allocation required={count}: {e}")))?;
    data.resize(count, T::default());
    Ok(data)
}
fn entry(candidate: &CandidateMeta) -> Entry {
    Entry {
        parent_idx: candidate.parent_idx,
        route: candidate.route,
        pad: 0,
    }
}
fn encode(entry: Entry, out: &mut [u8]) {
    out[..8].copy_from_slice(&entry.parent_idx.to_le_bytes());
    out[8..12].copy_from_slice(&entry.route.to_le_bytes());
    out[12..16].fill(0);
}
fn decode(bytes: [u8; 16]) -> Entry {
    Entry {
        parent_idx: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
        route: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        pad: 0,
    }
}
impl HistoryStore {
    pub fn new(config: HistoryConfig) -> Result<Self> {
        if config.capacity == 0 || config.slots == 0 || config.workers == 0 || config.moves == 0 {
            return Err(Error::Message(
                "history capacity/slots/workers/moves must be nonzero".into(),
            ));
        }
        fs::create_dir_all(&config.directory)?;
        let pinned = mul(mul(config.capacity as u64, config.slots as u64)?, 32)?;
        let chunk = config.capacity.min(1 << 20) as u64;
        let staging = mul(mul(chunk, config.slots as u64)?, 16)?;
        let mut stats = Stats {
            bytes_pinned_slots: pinned,
            bytes_slot_staging: staging,
            ..Default::default()
        };
        let mut plan = None;
        let mut ram: Vec<Entry> = Vec::new();
        let mut disk = None;
        if config.mode == HistoryMode::StaticHybrid {
            let overhead = pinned
                .checked_add(staging)
                .and_then(|v| v.checked_add(1))
                .ok_or(ContractError::Overflow("history RAM overhead"))?;
            require_capacity(
                "history pinned/staging RAM",
                overhead,
                config.ram_budget_bytes,
            )?;
            let ram_entries = (config.ram_budget_bytes - pinned - staging) / 16;
            let disk_entries = config.disk_budget_bytes / 16;
            let layout = history::plan(
                config.max_depth,
                config.capacity as u64,
                config.moves,
                ram_entries,
                disk_entries,
            )?;
            let estimate =
                history::estimate(config.max_depth, config.capacity as u64, config.moves)?;
            stats.required_entries = estimate.required_entries;
            mul(stats.required_entries, 16)?;
            stats.ram_reserved = layout.ram_reserved;
            stats.disk_reserved = layout.disk_reserved;
            stats.effective_depth = config.max_depth;
            stats.target_beam_depth = estimate.target_beam_depth;
            stats.states_before_target_beam = estimate.states_before_target_beam;
            stats.bytes_static_ram_arena = mul(ram_entries, 16)?;
            stats.bytes_static_disk_arena = mul(disk_entries, 16)?;
            ram = allocate(ram_entries)?;
            // Touch every writable page before search. Do not mlock a RAM arena.
            for index in (0..ram.len()).step_by(4096 / 16) {
                ram[index].pad = std::hint::black_box(0);
            }
            if disk_entries > 0 {
                let path = config
                    .disk_path
                    .clone()
                    .unwrap_or_else(|| config.directory.join("history_static_arena.bin"));
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(path)?;
                beam_native_sys::reserve_file(&file, stats.bytes_static_disk_arena)?;
                disk = Some(file);
            }
            plan = Some(layout);
        }
        let mut depths = Vec::new();
        depths
            .try_reserve_exact(config.max_depth as usize)
            .map_err(|e| Error::Message(e.to_string()))?;
        depths.resize_with(config.max_depth as usize, || None);
        // Reserve the same bounded staging budget as the native reference. The
        // serialized file writer uses its first chunk; no CandidateMeta copy.
        let staging = allocate(staging)?;
        Ok(Self {
            config,
            stats,
            plan,
            depths,
            ram,
            disk,
            disk_failed: false,
            staging,
        })
    }
    pub fn stats(&self) -> Stats {
        self.stats
    }
    pub fn count(&self, depth: u32) -> Result<u64> {
        self.depths
            .get(depth as usize)
            .and_then(Option::as_ref)
            .map(|d| d.count)
            .ok_or_else(|| Error::Message("history depth not materialized".into()))
    }
    fn write_file(
        staging: &mut [u8],
        file: &mut File,
        offset: u64,
        batch: &[CandidateMeta],
    ) -> Result<()> {
        file.seek(SeekFrom::Start(mul(offset, 16)?))?;
        for part in batch.chunks(staging.len() / 16) {
            for (candidate, out) in part.iter().zip(staging.chunks_exact_mut(16)) {
                encode(entry(candidate), out);
            }
            file.write_all(&staging[..part.len() * 16])?;
        }
        file.flush()?;
        Ok(())
    }
    fn write_ram(&mut self, offset: u64, batch: &[CandidateMeta]) -> Result<()> {
        let end = offset
            .checked_add(batch.len() as u64)
            .ok_or(ContractError::Overflow("history RAM end"))?;
        require_capacity("history RAM entries", end, self.ram.len() as u64)?;
        for (out, candidate) in self.ram[offset as usize..end as usize]
            .iter_mut()
            .zip(batch)
        {
            *out = entry(candidate);
        }
        Ok(())
    }
    pub fn write(&mut self, depth: u32, batch: &[CandidateMeta]) -> Result<()> {
        let slot = self
            .depths
            .get(depth as usize)
            .ok_or_else(|| Error::Message("history depth out of range".into()))?;
        if slot.is_some() {
            return Err(Error::Message("duplicate history depth".into()));
        }
        let count = batch.len() as u64;
        require_capacity("history batch", count, self.config.capacity as u64)?;
        let storage = match self.config.mode {
            HistoryMode::Ram => Storage::Ram(batch.iter().map(entry).collect()),
            HistoryMode::Disk => {
                let path = self
                    .config
                    .directory
                    .join(format!("depth_{depth}.candidate_meta.bin"));
                let mut file = File::create(&path)?;
                Self::write_file(&mut self.staging, &mut file, 0, batch)?;
                Storage::Disk(path)
            }
            HistoryMode::StaticHybrid => {
                let plan = self.plan.as_mut().unwrap();
                let reservation = plan.reservation(depth, count)?;
                let (location, mut offset) = (reservation.location, reservation.offset);
                let disk_result = if location == Location::Disk && !self.disk_failed {
                    Self::write_file(
                        &mut self.staging,
                        self.disk
                            .as_mut()
                            .ok_or_else(|| Error::Message("history disk arena missing".into()))?,
                        offset,
                        batch,
                    )
                } else {
                    Err(Error::Message("history disk disabled".into()))
                };
                if location == Location::Disk && disk_result.is_ok() {
                    Storage::ArenaDisk(offset)
                } else {
                    if location == Location::Disk {
                        self.disk_failed = true;
                        offset = plan.reserve_fallback(count).map_err(|error| {
                            Error::Message(format!("{}; {error}", disk_result.unwrap_err()))
                        })?;
                    }
                    self.write_ram(offset, batch)?;
                    Storage::ArenaRam(offset)
                }
            }
        };
        let stored = mul(count, 16)?;
        self.stats.bytes_received = self
            .stats
            .bytes_received
            .checked_add(mul(count, 32)?)
            .ok_or(ContractError::Overflow("history received bytes"))?;
        self.stats.bytes_stored = self
            .stats
            .bytes_stored
            .checked_add(stored)
            .ok_or(ContractError::Overflow("history stored bytes"))?;
        match &storage {
            Storage::Ram(_) | Storage::ArenaRam(_) => self.stats.bytes_stored_ram += stored,
            _ => self.stats.bytes_stored_disk += stored,
        }
        self.depths[depth as usize] = Some(Depth { count, storage });
        if self.config.mode == HistoryMode::Ram && self.config.prune {
            self.prune()?;
        }
        Ok(())
    }
    fn prune(&mut self) -> Result<()> {
        for index in (1..self.depths.len()).rev() {
            let (before, after) = self.depths.split_at_mut(index);
            let (Some(previous), Some(current)) = (&mut before[index - 1], &mut after[0]) else {
                continue;
            };
            let (Storage::Ram(prev), Storage::Ram(curr)) =
                (&mut previous.storage, &mut current.storage)
            else {
                continue;
            };
            let mut remap = vec![usize::MAX; prev.len()];
            for item in curr.iter() {
                let mapped = remap
                    .get_mut(
                        usize::try_from(item.parent_idx)
                            .map_err(|_| ContractError::Overflow("history parent"))?,
                    )
                    .ok_or_else(|| Error::Message("history prune parent out of range".into()))?;
                *mapped = 0;
            }
            let mut kept = 0;
            for (old, mapped) in remap.iter_mut().enumerate() {
                if *mapped != usize::MAX {
                    *mapped = kept;
                    prev[kept] = prev[old];
                    kept += 1;
                }
            }
            for item in curr {
                item.parent_idx = remap[item.parent_idx as usize] as u64;
            }
            let removed = (prev.len() - kept) as u64 * 16;
            prev.truncate(kept);
            previous.count = kept as u64;
            self.stats.bytes_stored -= removed;
            self.stats.bytes_stored_ram -= removed;
            self.stats.bytes_pruned += removed;
        }
        Ok(())
    }
    pub fn read(&mut self, depth: u32, index: u64) -> Result<Entry> {
        let count = self.count(depth)?;
        if index >= count {
            return Err(Error::Message(format!(
                "history index out of range: index={index} count={count}"
            )));
        }
        let record = self.depths[depth as usize].as_ref().unwrap();
        let mut bytes = [0; 16];
        match &record.storage {
            Storage::Ram(data) => Ok(data[index as usize]),
            Storage::ArenaRam(offset) => Ok(self.ram[(*offset + index) as usize]),
            Storage::Disk(path) => {
                let mut file = File::open(path)?;
                file.seek(SeekFrom::Start(mul(index, 16)?))?;
                file.read_exact(&mut bytes)?;
                Ok(decode(bytes))
            }
            Storage::ArenaDisk(offset) => {
                let file = self
                    .disk
                    .as_mut()
                    .ok_or_else(|| Error::Message("history disk missing".into()))?;
                file.seek(SeekFrom::Start(mul(*offset + index, 16)?))?;
                file.read_exact(&mut bytes)?;
                Ok(decode(bytes))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disk_error_fallback_cannot_overwrite_future_ram_layer() {
        let directory =
            std::env::temp_dir().join(format!("beam-history-fallback-{}", std::process::id()));
        let config = HistoryConfig {
            mode: HistoryMode::StaticHybrid,
            capacity: 8,
            slots: 2,
            workers: 2,
            max_depth: 4,
            moves: 2,
            ram_budget_bytes: 928,
            disk_budget_bytes: 14 * 16,
            directory: directory.clone(),
            disk_path: None,
            prune: false,
        };
        let mut store = HistoryStore::new(config).unwrap();
        // Replace only this test's owned file with a read-only descriptor to
        // inject a real write error even when tests run as root in a container.
        store.disk = Some(File::open(directory.join("history_static_arena.bin")).unwrap());
        let old = CandidateMeta {
            route: 77,
            ..Default::default()
        };
        let future = CandidateMeta {
            route: 88,
            ..Default::default()
        };
        store.write(0, &[old; 2]).unwrap();
        store.write(3, &[future; 8]).unwrap();
        assert_eq!(store.read(0, 1).unwrap().route, 77);
        assert_eq!(store.read(3, 7).unwrap().route, 88);
        let error = store.write(1, &[old]).unwrap_err().to_string();
        assert!(
            error.contains("required=11") && error.contains("available=10"),
            "{error}"
        );
        assert_eq!(store.stats.bytes_stored_ram, 160);
        assert_eq!(store.stats.bytes_stored_disk, 0);
        drop(store);
        fs::remove_dir_all(directory).unwrap();
    }
}
