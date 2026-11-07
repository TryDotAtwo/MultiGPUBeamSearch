use beam_core::{
    capacity::{frontier_ceiling, BeamWidth},
    search::PermutationGraph,
};

#[test]
fn finite_multiset_bound_excludes_unique_inverse_neighbors() {
    let graph = PermutationGraph {
        center: vec![0, 0, 1, 1],
        generators: vec![vec![1, 0, 2, 3], vec![2, 1, 0, 3], vec![2, 1, 0, 3]],
    };
    // Six distinct arrangements; center and one distinct inverse neighbor.
    let bound = frontier_ceiling(&graph, 2, 16, &[1024, 2048], None).unwrap();
    assert_eq!(bound.upper_bound, 4);
    assert_eq!(bound.excluded_states, 2);
    assert!(!bound.reachable_orbit_verified);
    assert_eq!(
        frontier_ceiling(&graph, 2, 16, &[1024, 2048], Some(3))
            .unwrap()
            .upper_bound,
        3
    );
    assert!(frontier_ceiling(&graph, 2, 16, &[1024], None).is_err());
}

#[test]
fn large_factorial_is_clamped_without_overflow_or_false_reachability() {
    let graph = PermutationGraph {
        center: (0..150).collect(),
        generators: vec![(0..150).rev().collect()],
    };
    let bound = frontier_ceiling(&graph, 2, 160, &[u64::MAX, u64::MAX], None).unwrap();
    assert_eq!(bound.upper_bound, ((u64::MAX as u128 * 2) / 160) as u64);
    assert!(bound.multiset_bound_clamped);
    assert!(!bound.reachable_orbit_verified);
}

#[test]
fn width_mode_is_exact_integer_or_explicit_maximum() {
    for bad in ["0", "-1", "1.5", "true", "\"auto\"", "\"1024\""] {
        assert!(serde_json::from_str::<BeamWidth>(bad).is_err(), "{bad}");
    }
    assert_eq!(
        serde_json::from_str::<BeamWidth>("4097").unwrap(),
        BeamWidth::Exact(4097)
    );
    assert_eq!(
        serde_json::from_str::<BeamWidth>("\"max\"").unwrap(),
        BeamWidth::Maximum
    );
}
