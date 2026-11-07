//! Typed, CUDA-independent contracts shared by native launchers and adapters.
#![forbid(unsafe_code)]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
pub mod history;

#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("invalid {0}")]
    Invalid(&'static str),
    #[error("{0} size overflow")]
    Overflow(&'static str),
    #[error("{name} capacity exceeded: required={required} available={available}")]
    Capacity {
        name: String,
        required: u64,
        available: u64,
    },
}
pub fn require_capacity(name: &str, required: u64, available: u64) -> Result<(), ContractError> {
    if required > available {
        Err(ContractError::Capacity {
            name: name.into(),
            required,
            available,
        })
    } else {
        Ok(())
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Hash128 {
    pub lo: u64,
    pub hi: u64,
}
impl Ord for Hash128 {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.hi, self.lo).cmp(&(other.hi, other.lo))
    }
}
impl PartialOrd for Hash128 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default)]
pub struct CandidateMeta {
    pub hash: Hash128,
    pub parent_idx: u64,
    pub score: u32,
    pub route: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Domain {
    Permutation {
        logical_bytes: u64,
        moves: u32,
        alignment: u32,
    },
    Cnot {
        wires: u32,
    },
}
impl Domain {
    pub fn permutation(
        logical_bytes: u64,
        moves: u32,
        alignment: u32,
    ) -> Result<Self, ContractError> {
        let value = Self::Permutation {
            logical_bytes,
            moves,
            alignment,
        };
        value.state_bytes()?;
        Ok(value)
    }
    pub fn cnot(wires: u32) -> Result<Self, ContractError> {
        let value = Self::Cnot { wires };
        value.state_bytes()?;
        Ok(value)
    }
    pub fn state_bytes(&self) -> Result<u64, ContractError> {
        match *self {
            Self::Permutation {
                logical_bytes,
                moves,
                alignment,
            } => {
                if logical_bytes == 0
                    || !(1..=255).contains(&moves)
                    || !matches!(alignment, 16 | 32 | 64)
                {
                    return Err(ContractError::Invalid("permutation geometry"));
                }
                // Four temporary FinalResponse bytes must fit in the padding.
                logical_bytes
                    .checked_add(4)
                    .and_then(|n| n.checked_add(alignment as u64 - 1))
                    .map(|n| n & !(alignment as u64 - 1))
                    .ok_or(ContractError::Overflow("permutation state"))
            }
            Self::Cnot { wires } => {
                if !(2..=256).contains(&wires) {
                    return Err(ContractError::Invalid("CNOT wires"));
                }
                Ok(wires as u64 * (wires as u64).div_ceil(64) * 8)
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rank {
    pub world: u32,
    pub global: u32,
    pub local_world: u32,
    pub local: u32,
}
impl Rank {
    pub fn new(
        world: u32,
        global: u32,
        local_world: u32,
        local: u32,
    ) -> Result<Self, ContractError> {
        let value = Self {
            world,
            global,
            local_world,
            local,
        };
        value.validate()?;
        Ok(value)
    }
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.world == 0
            || self.local_world == 0
            || self.local_world > self.world
            || self.global >= self.world
            || self.local >= self.local_world
        {
            Err(ContractError::Invalid("rank placement"))
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SearchConfig {
    pub domain: Domain,
    pub beam_width: u64,
    pub max_depth: u32,
    pub rank: Rank,
}
impl SearchConfig {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.domain.state_bytes()?;
        self.rank.validate()?;
        if self.beam_width == 0 || self.max_depth == 0 {
            return Err(ContractError::Invalid("search width/depth"));
        }
        self.rank_frontier_capacity()?
            .checked_mul(self.domain.state_bytes()?)
            .ok_or(ContractError::Overflow("frontier"))?;
        Ok(())
    }
    pub fn rank_frontier_capacity(&self) -> Result<u64, ContractError> {
        self.rank.validate()?;
        Ok(self.beam_width.div_ceil(self.rank.world as u64))
    }
}
