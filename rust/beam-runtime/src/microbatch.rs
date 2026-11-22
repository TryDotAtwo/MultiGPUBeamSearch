//! Existing native inference microbatch capacity planning.
//!
//! This is a policy transfer only: retain the established integer formula and
//! expose its result to the Python runner adapter without changing search.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

const ALIGNMENT: u128 = 1024;
const INITIAL_SORT_SLOTS: u128 = 4;
const NATIVE_DEFAULT_ROW_BUDGET: u64 = 8192;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub beam_width: u64,
    pub world_size: u32,
    pub move_count: u32,
    pub output_dim: u32,
    pub model_backend: String,
    pub row_limit: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub native_default_row_budget: u64,
    pub configured_row_budget: u64,
    pub derived_parent_batch: u64,
    pub derived_candidates_per_slot: u64,
    pub reference_logical_shard: u64,
    pub reference_shard_storage: u64,
    pub adjusted_for_small_beam: bool,
}

fn invalid(message: &str) -> Error {
    Error::Message(message.into())
}

fn as_u64(value: u128) -> Result<u64> {
    u64::try_from(value).map_err(|_| invalid("microbatch plan exceeds uint64 range"))
}

/// Preserve the Python planner's exact integer choices and metadata fields.
pub fn resolve(request: &Request) -> Result<Plan> {
    if request.beam_width == 0 {
        return Err(invalid("beam_width must be positive for microbatch planning"));
    }
    if request.world_size == 0 {
        return Err(invalid("world_size must be positive for microbatch planning"));
    }
    if request.move_count == 0 {
        return Err(invalid("move_count must be positive for microbatch planning"));
    }
    if request.output_dim == 0 {
        return Err(invalid("output_dim must be positive for microbatch planning"));
    }
    if request.row_limit == 0 {
        return Err(invalid("inference row limit must be positive for microbatch planning"));
    }

    let divisor = u128::from(request.world_size) * INITIAL_SORT_SLOTS * ALIGNMENT;
    let beam = u128::from(request.beam_width);
    let logical_shard = ((beam + divisor - 1) / divisor) * ALIGNMENT;
    let scaled_capacity = (logical_shard * 5 + 3) / 4;
    let shard_storage = ((scaled_capacity + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    let rows_per_parent = if request.output_dim == 1 && request.model_backend != "ensemble" {
        u128::from(request.move_count)
    } else {
        1
    };
    let default_rows = u128::from(request.row_limit);
    let default_parents = default_rows / rows_per_parent;
    let parent_batch = default_parents.min(shard_storage / u128::from(request.move_count));
    let row_budget = if parent_batch == default_parents {
        default_rows
    } else {
        parent_batch * rows_per_parent
    };
    let candidate_rows = parent_batch * u128::from(request.move_count);

    Ok(Plan {
        native_default_row_budget: request.row_limit,
        configured_row_budget: as_u64(row_budget)?,
        derived_parent_batch: as_u64(parent_batch)?,
        derived_candidates_per_slot: as_u64(candidate_rows)?,
        reference_logical_shard: as_u64(logical_shard)?,
        reference_shard_storage: as_u64(shard_storage)?,
        adjusted_for_small_beam: row_budget != u128::from(NATIVE_DEFAULT_ROW_BUDGET),
    })
}

#[cfg(test)]
mod tests {
    use super::{resolve, Plan, Request};

    fn request(beam_width: u64, world_size: u32, move_count: u32, output_dim: u32,
               model_backend: &str, row_limit: u64) -> Request {
        Request {
            beam_width,
            world_size,
            move_count,
            output_dim,
            model_backend: model_backend.into(),
            row_limit,
        }
    }

    #[test]
    fn preserves_scalar_head_parent_alignment_and_capacity() {
        assert_eq!(
            resolve(&request(1, 2, 12, 1, "mlp", 8192)).unwrap(),
            Plan {
                native_default_row_budget: 8192,
                configured_row_budget: 2040,
                derived_parent_batch: 170,
                derived_candidates_per_slot: 2040,
                reference_logical_shard: 1024,
                reference_shard_storage: 2048,
                adjusted_for_small_beam: true,
            }
        );
    }

    #[test]
    fn preserves_q_head_ensemble_and_nondefault_row_limit() {
        assert_eq!(
            resolve(&request(20000, 1, 4, 1, "ensemble", 1234)).unwrap(),
            Plan {
                native_default_row_budget: 1234,
                configured_row_budget: 1234,
                derived_parent_batch: 1234,
                derived_candidates_per_slot: 4936,
                reference_logical_shard: 5120,
                reference_shard_storage: 7168,
                adjusted_for_small_beam: true,
            }
        );
        assert_eq!(
            resolve(&request(100000, 1, 4, 4, "mlp", 8192)).unwrap(),
            Plan {
                native_default_row_budget: 8192,
                configured_row_budget: 8192,
                derived_parent_batch: 8192,
                derived_candidates_per_slot: 32768,
                reference_logical_shard: 25600,
                reference_shard_storage: 32768,
                adjusted_for_small_beam: false,
            }
        );
    }

    #[test]
    fn rejects_zero_geometry_and_budget() {
        assert!(resolve(&request(0, 1, 4, 4, "mlp", 8192)).is_err());
        assert!(resolve(&request(1, 0, 4, 4, "mlp", 8192)).is_err());
        assert!(resolve(&request(1, 1, 0, 4, "mlp", 8192)).is_err());
        assert!(resolve(&request(1, 1, 4, 0, "mlp", 8192)).is_err());
        assert!(resolve(&request(1, 1, 4, 4, "mlp", 0)).is_err());
    }
}
