//! CPU-testable, shared panel-v2 manifest identity contract.
//! The CUDA bridge consumes this same typed manifest and validator.

use super::manifest::PinnedArtifact;
use super::panel::frozen_panel_digest;
use crate::device_search::{ActionHead, BeamConfig};
use serde::Deserialize;
use std::collections::BTreeSet;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub(super) schema: String,
    pub(super) training_manifest: PinnedArtifact,
    pub(super) training_result: PinnedArtifact,
    pub(super) panel: String,
    pub(super) panel_indices: Vec<usize>,
    pub(super) panel_ids: Vec<String>,
    pub(super) panel_digest: String,
    #[serde(rename = "search")]
    pub(super) _search: BeamConfig,
    #[serde(rename = "head")]
    pub(super) _head: ActionHead,
}

pub(super) fn validate_manifest_identity(spec: &Manifest) -> Result<(), String> {
    if spec.schema != "cnot103_panel_search_v2"
        || !matches!(spec.panel.as_str(), "dev" | "test")
        || spec.panel_indices.is_empty()
        || spec.panel_indices.len() != spec.panel_ids.len()
        || spec.panel_indices.iter().collect::<BTreeSet<_>>().len() != spec.panel_indices.len()
        || spec.panel_ids.iter().any(|id| id.trim().is_empty())
        || spec.panel_ids.iter().collect::<BTreeSet<_>>().len() != spec.panel_ids.len()
        || frozen_panel_digest(&spec.panel, &spec.panel_indices, &spec.panel_ids)
            != spec.panel_digest
    {
        return Err("panel-v2 manifest identity is invalid".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn valid_manifest() -> Manifest {
        let panel = "dev".to_string();
        let panel_indices = vec![2usize, 5usize];
        let panel_ids = vec!["case-a".to_string(), "case-b".to_string()];
        let panel_digest = frozen_panel_digest(&panel, &panel_indices, &panel_ids);
        serde_json::from_value(json!({
            "schema": "cnot103_panel_search_v2",
            "training_manifest": {"path": "run.json", "sha256": "a".repeat(64)},
            "training_result": {"path": "training-result.json", "sha256": "b".repeat(64)},
            "panel": panel,
            "panel_indices": panel_indices,
            "panel_ids": panel_ids,
            "panel_digest": panel_digest,
            "search": {
                "beam_width": 36,
                "shortlist": 4,
                "max_depth": 8,
                "max_expanded": 201,
                "inference_batch": 32
            },
            "head": "qhat"
        }))
        .expect("synthetic panel-v2 manifest parses")
    }

    #[test]
    fn accepts_a_digest_bound_ordered_panel_manifest() {
        let spec = valid_manifest();
        validate_manifest_identity(&spec).expect("valid panel manifest");
    }

    #[test]
    fn rejects_legacy_schema() {
        let mut spec = valid_manifest();
        spec.schema = "cnot103_dev_search_v1".into();
        assert!(validate_manifest_identity(&spec).is_err());
    }

    #[test]
    fn rejects_duplicate_indices_and_ids() {
        let mut spec = valid_manifest();
        spec.panel_indices[1] = spec.panel_indices[0];
        assert!(validate_manifest_identity(&spec).is_err());

        let mut spec = valid_manifest();
        spec.panel_ids[1] = spec.panel_ids[0].clone();
        assert!(validate_manifest_identity(&spec).is_err());
    }

    #[test]
    fn rejects_reordered_or_wrong_digest() {
        let mut spec = valid_manifest();
        spec.panel_indices.reverse();
        spec.panel_ids.reverse();
        assert!(validate_manifest_identity(&spec).is_err());

        let mut spec = valid_manifest();
        spec.panel_digest = "0".repeat(64);
        assert!(validate_manifest_identity(&spec).is_err());
    }
}
