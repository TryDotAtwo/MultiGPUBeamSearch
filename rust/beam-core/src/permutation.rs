//! CPU task contracts and independent path replay for permutation search.
use crate::ContractError;
use serde::{Deserialize, Serialize};
type Result<T> = std::result::Result<T, ContractError>;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermutationGraph {
    pub center: Vec<u8>,
    pub generators: Vec<Vec<u32>>,
}
impl PermutationGraph {
    pub fn validate(&self) -> Result<()> {
        let size = self.center.len();
        if !(1..=150).contains(&size) || !(1..=255).contains(&self.generators.len()) {
            return Err(ContractError::Invalid("permutation graph geometry"));
        }
        let limit = if size > 128 { 255 } else { 127 };
        if self.center.iter().any(|v| *v > limit) {
            return Err(ContractError::Invalid("state labels"));
        }
        for generator in &self.generators {
            if generator.len() != size {
                return Err(ContractError::Invalid("generator length"));
            }
            let mut seen = vec![false; size];
            for source in generator {
                let slot = seen
                    .get_mut(*source as usize)
                    .ok_or(ContractError::Invalid("generator index"))?;
                if *slot {
                    return Err(ContractError::Invalid("generator is not a permutation"));
                }
                *slot = true;
            }
        }
        Ok(())
    }
    pub fn validate_start(&self, start: &[u8]) -> Result<()> {
        self.validate()?;
        let mut expected = self.center.clone();
        let mut actual = start.to_vec();
        expected.sort_unstable();
        actual.sort_unstable();
        if actual != expected {
            return Err(ContractError::Invalid("start state multiset"));
        }
        Ok(())
    }
    pub fn replay(&self, start: &[u8], moves: &[u32]) -> Result<bool> {
        self.validate_start(start)?;
        let mut state = start.to_vec();
        let mut next = vec![0; state.len()];
        for movement in moves {
            let generator = self
                .generators
                .get(*movement as usize)
                .ok_or(ContractError::Invalid("solution generator IDs"))?;
            for (target, source) in next.iter_mut().zip(generator) {
                *target = state[*source as usize];
            }
            std::mem::swap(&mut state, &mut next);
        }
        Ok(state == self.center)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchBudget {
    pub requested_max_steps: u32,
    pub native_forward_depth_limit: u32,
    pub configured_touch_bfs_radius: u32,
    pub effective_touch_bfs_radius: u32,
    pub touch_bfs_max_entries: u64,
    pub touch_bfs_worst_case_entries: u64,
}
pub fn search_budget(
    max_steps: u32,
    radius: u32,
    moves: u32,
    max_entries: u64,
) -> Result<SearchBudget> {
    if moves == 0 || moves > 255 || radius > 12 || max_entries == 0 {
        return Err(ContractError::Invalid("touch BFS configuration"));
    }
    let effective_radius = radius.min(max_steps.saturating_sub(1));
    if effective_radius > 0 && moves > 32 {
        return Err(ContractError::Invalid("touch BFS suffix move packing"));
    }
    let mut total = 1u64;
    let mut frontier = 1u64;
    for _ in 0..effective_radius {
        frontier = frontier
            .checked_mul(moves as u64)
            .ok_or(ContractError::Overflow("touch BFS frontier"))?;
        total = total
            .checked_add(frontier)
            .ok_or(ContractError::Overflow("touch BFS entries"))?;
        crate::require_capacity("touch BFS entries", total, max_entries)?;
    }
    Ok(SearchBudget {
        requested_max_steps: max_steps,
        native_forward_depth_limit: max_steps - effective_radius,
        configured_touch_bfs_radius: radius,
        effective_touch_bfs_radius: effective_radius,
        touch_bfs_max_entries: max_entries,
        touch_bfs_worst_case_entries: total,
    })
}
