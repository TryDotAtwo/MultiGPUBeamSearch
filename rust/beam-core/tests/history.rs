use beam_core::history::{estimate, plan, Location};

#[test]
fn exact_history_reservations_and_tail_fallback() {
    let estimate = estimate(5, 100, 3).unwrap();
    assert_eq!(
        (
            estimate.target_beam_depth,
            estimate.states_before_target_beam,
            estimate.required_entries
        ),
        (4, 120, 220)
    );
    let mut plan = plan(5, 100, 3, 200, 120).unwrap();
    assert_eq!((plan.ram_reserved, plan.disk_reserved), (100, 120));
    assert_eq!(
        (plan.depths[3].location, plan.depths[3].offset),
        (Location::Disk, 39)
    );
    assert_eq!(
        (plan.depths[4].location, plan.depths[4].offset),
        (Location::Ram, 0)
    );
    // Smaller actual batches do not change later reservations. Disk fallback
    // starts after all future planned RAM layers, never at the last write.
    assert!(plan.reservation(3, 2).is_ok());
    assert_eq!(plan.reserve_fallback(81).unwrap(), 100);
    assert_eq!(plan.reserve_fallback(19).unwrap(), 181);
    let error = plan.reserve_fallback(1).unwrap_err().to_string();
    assert!(error.contains("required=201") && error.contains("available=200"));
}

#[test]
fn full_depth_placement_zero_and_arithmetic_limits() {
    assert_eq!(estimate(0, 100, 3).unwrap().required_entries, 0);
    assert_eq!(estimate(9, 0, 3).unwrap().required_entries, 0);
    assert!(estimate(1, 1, 0).is_err());
    assert!(estimate(u32::MAX, u64::MAX, u32::MAX).is_err());
    assert_eq!(
        estimate(u32::MAX, 100, 1).unwrap().required_entries,
        u32::MAX as u64
    );
    assert!(plan(2, 4, 4, 3, 5).is_err()); // Total fits, whole second layer does not.
    let p = plan(3, 10, 3, u64::MAX, u64::MAX).unwrap();
    assert_eq!(p.required, 22);
    assert!(p
        .reservation(0, 4)
        .unwrap_err()
        .to_string()
        .contains("required=4"));
    assert!(p.reservation(3, 0).is_err());
}
