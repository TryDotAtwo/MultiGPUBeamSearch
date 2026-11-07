use beam_core::{require_capacity, CandidateMeta, Domain, Hash128, Rank, SearchConfig};
use std::mem::{align_of, size_of};

#[test]
fn binary_layout_and_complete_hash_order() {
    assert_eq!(
        (size_of::<CandidateMeta>(), align_of::<CandidateMeta>()),
        (32, 32)
    );
    assert_eq!(size_of::<Hash128>(), 16);
    assert!(
        Hash128 {
            lo: u64::MAX,
            hi: 0
        } < Hash128 { lo: 0, hi: 1 }
    );
    assert_ne!(Hash128 { lo: 7, hi: 1 }, Hash128 { lo: 7, hi: 2 });
    assert_ne!(Hash128 { lo: 1, hi: 7 }, Hash128 { lo: 2, hi: 7 });
}

#[test]
fn domains_do_not_confuse_matrix_words_and_permutation_padding() {
    let cube = Domain::permutation(96, 24, 16).unwrap();
    assert_eq!(cube.state_bytes().unwrap(), 112);
    assert_eq!(
        Domain::permutation(120, 24, 16)
            .unwrap()
            .state_bytes()
            .unwrap(),
        128
    );
    assert_eq!(Domain::cnot(144).unwrap().state_bytes().unwrap(), 3456);
    assert_eq!(Domain::cnot(256).unwrap().state_bytes().unwrap(), 8192);
    assert!(Domain::cnot(257).is_err());
    assert!(Domain::permutation(u64::MAX, 24, 16).is_err());
    assert!(Domain::permutation(96, 24, 3).is_err());
    assert!(Domain::permutation(96, 24, 8).is_err());
    assert!(Domain::permutation(96, 256, 16).is_err());
}

#[test]
fn exact_global_width_and_overflow_diagnostics() {
    let cfg = SearchConfig {
        domain: Domain::permutation(96, 24, 16).unwrap(),
        beam_width: 10,
        max_depth: 3,
        rank: Rank::new(3, 1, 3, 1).unwrap(),
    };
    cfg.validate().unwrap();
    assert_eq!(cfg.rank_frontier_capacity().unwrap(), 4);
    assert_eq!(cfg.beam_width, 10);
    assert!(Rank::new(2, 2, 2, 0).is_err());
    assert!(Rank::new(2, 0, 1, 1).is_err());
    assert_eq!(
        require_capacity("history", 33, 32).unwrap_err().to_string(),
        "history capacity exceeded: required=33 available=32"
    );
    assert!(require_capacity("history", 0, 0).is_ok());
}
