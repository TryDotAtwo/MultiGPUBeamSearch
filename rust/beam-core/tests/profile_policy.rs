use beam_core::{
    components::ServiceEnvelope,
    profile_policy::{proposals, Hardware, SearchGeometry},
};
use std::collections::{BTreeMap, BTreeSet};
#[test]
fn proposals_preserve_frozen_inference_and_semantic_controls() {
    let baseline: BTreeMap<String, String> = [
        ("BEAM_B_MICRO", "384"),
        ("BEAM_SHARD_COUNT", "2"),
        ("BEAM_STREAM3_RING_SLOTS", "4"),
        ("BEAM_STREAM4_ACTIVE_SORT_SLOTS", "2"),
        ("BEAM_STREAM4_BATCH_CANDIDATES", "1024"),
        ("BEAM_ENSEMBLE_INFERENCE_MICRO", "16"),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), v.into()))
    .collect();
    let geometry = SearchGeometry {
        beam: 65537,
        world: 2,
        moves: 24,
        alignment: 1024,
        row_factor: 24,
    };
    let hardware = Hardware {
        sm_count: 28,
        l2_bytes: 3 << 20,
    };
    let service = ServiceEnvelope {
        service_envelope_seconds: 1.,
        service_work_seconds: 1.03,
        downstream_service_work_seconds: 0.03,
        inference_seconds: 1.,
        stream3_seconds: 0.01,
        stream4_seconds: 0.01,
        transport_seconds: 0.005,
        union_seconds: 0.005,
        scope: "proxy".into(),
    };
    let p = proposals(
        &geometry,
        &baseline,
        &hardware,
        &service,
        false,
        &BTreeSet::new(),
    )
    .unwrap();
    assert_eq!(p.regime, "inference_dominated");
    assert!(!p.candidates.is_empty() && p.candidates.len() <= 4);
    assert!(p.candidates[0].name.starts_with("outer-"));
    let mut seen = BTreeSet::new();
    for row in p.candidates {
        assert_eq!(row.environment["BEAM_ENSEMBLE_INFERENCE_MICRO"], "16");
        assert_eq!(
            row.environment.keys().collect::<Vec<_>>(),
            baseline.keys().collect::<Vec<_>>()
        );
        assert!(row.environment["BEAM_B_MICRO"].parse::<u64>().unwrap() % 384 == 0);
        assert!(
            row.environment["BEAM_STREAM4_BATCH_CANDIDATES"]
                .parse::<u64>()
                .unwrap()
                >= 1024
        );
        assert_ne!(row.environment, baseline);
        assert!(seen.insert(row.environment));
    }
    let fine = proposals(&geometry, &baseline, &hardware, &service, true, &seen).unwrap();
    assert!(fine.candidates.len() <= 10);
    assert!(fine
        .candidates
        .iter()
        .all(|r| !seen.contains(&r.environment)));
    let mut invalid = baseline.clone();
    invalid.insert("BEAM_SOLVED_NEIGHBORHOOD_RADIUS".into(), "0".into());
    assert!(proposals(&geometry, &invalid, &hardware, &service, false, &seen).is_err());
    invalid = baseline.clone();
    invalid.insert("BEAM_B_MICRO".into(), "383".into());
    assert!(proposals(&geometry, &invalid, &hardware, &service, false, &seen).is_err());
}
