//! Whole-layer history reservations. All quantities are logical entry counts.
use crate::{require_capacity, ContractError};
type Result<T> = std::result::Result<T, ContractError>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub parent_idx: u64,
    pub route: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub bytes_received: u64,
    pub bytes_stored: u64,
    pub bytes_stored_ram: u64,
    pub bytes_stored_disk: u64,
    pub bytes_pruned: u64,
    pub bytes_pinned_slots: u64,
    pub bytes_slot_staging: u64,
    pub bytes_static_ram_arena: u64,
    pub bytes_static_disk_arena: u64,
    pub required_entries: u64,
    pub ram_reserved: u64,
    pub disk_reserved: u64,
    pub states_before_target_beam: u64,
    pub effective_depth: u32,
    pub target_beam_depth: u32,
}

#[derive(Debug, Default)]
pub struct Estimate {
    pub required_entries: u64,
    pub target_beam_depth: u32,
    pub states_before_target_beam: u64,
}

fn expand(previous: u64, beam: u64, moves: u32) -> u64 {
    if previous > beam / moves as u64 {
        beam
    } else {
        beam.min(previous * moves as u64)
    }
}
pub fn estimate(depths: u32, beam: u64, moves: u32) -> Result<Estimate> {
    if moves == 0 {
        return Err(ContractError::Invalid("history move count"));
    }
    if depths == 0 || beam == 0 {
        return Ok(Estimate::default());
    }
    if moves == 1 && beam > 1 {
        return Ok(Estimate {
            required_entries: depths as u64,
            target_beam_depth: depths,
            states_before_target_beam: depths as u64,
        });
    }
    let mut result = Estimate::default();
    let mut bound = 1;
    for depth in 0..depths {
        bound = expand(bound, beam, moves);
        if bound == beam {
            result.target_beam_depth = depth;
            result.required_entries = (depths - depth as u32) as u64;
            result.required_entries = result
                .required_entries
                .checked_mul(beam)
                .and_then(|v| v.checked_add(result.states_before_target_beam))
                .ok_or(ContractError::Overflow("history required entries"))?;
            return Ok(result);
        }
        result.states_before_target_beam = result
            .states_before_target_beam
            .checked_add(bound)
            .ok_or(ContractError::Overflow("history prefull entries"))?;
    }
    result.target_beam_depth = depths;
    result.required_entries = result.states_before_target_beam;
    Ok(result)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Location {
    Ram,
    Disk,
}
#[derive(Debug)]
pub struct Reservation {
    pub bound: u64,
    pub offset: u64,
    pub location: Location,
}
#[derive(Debug)]
pub struct Plan {
    pub depths: Vec<Reservation>,
    pub required: u64,
    pub ram_reserved: u64,
    pub disk_reserved: u64,
    ram_capacity: u64,
    fallback_end: u64,
}
impl Plan {
    pub fn reservation(&self, depth: u32, actual: u64) -> Result<&Reservation> {
        let r = self
            .depths
            .get(depth as usize)
            .ok_or(ContractError::Invalid("history depth"))?;
        require_capacity("history layer", actual, r.bound)?;
        Ok(r)
    }
    pub fn reserve_fallback(&mut self, count: u64) -> Result<u64> {
        let required = self
            .fallback_end
            .checked_add(count)
            .ok_or(ContractError::Overflow("history fallback"))?;
        require_capacity("history RAM fallback", required, self.ram_capacity)?;
        let offset = self.fallback_end;
        self.fallback_end = required;
        Ok(offset)
    }
}
pub fn plan(depths: u32, beam: u64, moves: u32, ram: u64, disk: u64) -> Result<Plan> {
    let required = estimate(depths, beam, moves)?.required_entries;
    require_capacity("history total entries", required, ram.saturating_add(disk))?;
    let mut p = Plan {
        depths: Vec::new(),
        required,
        ram_reserved: 0,
        disk_reserved: 0,
        ram_capacity: ram,
        fallback_end: 0,
    };
    p.depths
        .try_reserve_exact(depths as usize)
        .map_err(|_| ContractError::Invalid("history depth allocation"))?;
    let mut bound = if beam == 0 { 0 } else { 1 };
    for _ in 0..depths {
        bound = expand(bound, beam, moves);
        if bound <= disk - p.disk_reserved {
            p.depths.push(Reservation {
                bound,
                offset: p.disk_reserved,
                location: Location::Disk,
            });
            p.disk_reserved += bound;
        } else if bound <= ram - p.ram_reserved {
            p.depths.push(Reservation {
                bound,
                offset: p.ram_reserved,
                location: Location::Ram,
            });
            p.ram_reserved += bound;
        } else {
            return Err(ContractError::Capacity {
                name: "history whole layer".into(),
                required: bound,
                available: (ram - p.ram_reserved).max(disk - p.disk_reserved),
            });
        }
    }
    p.fallback_end = p.ram_reserved;
    Ok(p)
}
