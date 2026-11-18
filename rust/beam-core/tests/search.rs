use beam_core::search::{
    native_depth_budget, search_budget, touch_bfs_worst_case_entries,
    validate_public_search_controls, PermutationGraph, PublicSearchControls,
    SearchControlError, TerminalParser,
};

fn public_controls(
    beam_width: serde_json::Value,
    max_steps: serde_json::Value,
) -> PublicSearchControls {
    serde_json::from_value(serde_json::json!({
        "beam_width": beam_width,
        "max_steps": max_steps,
        "beam_mode": {"kind": "str", "value": "simple"},
        "history_depth": {"kind": "int", "value": "0"},
        "destination_state_present": false,
        "bfs_result_for_mitm_present": false,
        "return_path": {"kind": "bool", "value": true},
        "use_child_scores": {"kind": "bool", "value": false}
    }))
    .unwrap()
}

#[test]
fn public_search_controls_accept_fixed_and_maximum_width_requests() {
    let fixed = public_controls(json_tag("int", "64"), json_tag("int", "8"));
    assert_eq!(validate_public_search_controls(&fixed), Ok(()));
    let maximum = public_controls(json_tag("str", "max"), json_tag("int", "0"));
    assert_eq!(validate_public_search_controls(&maximum), Ok(()));
    let mut false_history = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    false_history.history_depth = serde_json::from_value(serde_json::json!({
        "kind": "bool",
        "value": false
    }))
    .unwrap();
    assert_eq!(validate_public_search_controls(&false_history), Ok(()));
    let mut zero_float_history = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    zero_float_history.history_depth = serde_json::from_value(json_tag("float", "0.0")).unwrap();
    assert_eq!(validate_public_search_controls(&zero_float_history), Ok(()));
}

#[test]
fn public_search_controls_preserve_invalid_and_unsupported_contracts() {
    let invalid_width = public_controls(json_tag("int", "0"), json_tag("int", "4"));
    assert_eq!(
        validate_public_search_controls(&invalid_width),
        Err(SearchControlError::Invalid("beam_width must be an integer > 0".into()))
    );
    let oversized_depth = public_controls(
        json_tag("int", "1"),
        json_tag("int", "4294967296"),
    );
    assert_eq!(
        validate_public_search_controls(&oversized_depth),
        Err(SearchControlError::Invalid(
            "max_steps exceeds native integer range (4294967295)".into()
        ))
    );
    let oversized_width = public_controls(
        json_tag("int", "18446744073709551616"),
        json_tag("int", "4"),
    );
    assert_eq!(
        validate_public_search_controls(&oversized_width),
        Err(SearchControlError::Invalid(
            "beam_width exceeds native integer range (18446744073709551615)".into()
        ))
    );
    let boolean_width = public_controls(
        serde_json::json!({"kind": "bool", "value": true}),
        json_tag("int", "4"),
    );
    assert_eq!(
        validate_public_search_controls(&boolean_width),
        Err(SearchControlError::Invalid("beam_width must be an integer > 0".into()))
    );
    let mut advanced = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    advanced.beam_mode = serde_json::from_value(json_tag("str", "advanced")).unwrap();
    assert_eq!(
        validate_public_search_controls(&advanced),
        Err(SearchControlError::Unsupported(
            "native does not implement the upstream advanced mode; use beam_mode='simple'".into()
        ))
    );
    let mut taboo = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    taboo.history_depth = serde_json::from_value(json_tag("int", "1")).unwrap();
    assert!(matches!(
        validate_public_search_controls(&taboo),
        Err(SearchControlError::Unsupported(_))
    ));
    let mut destination = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    destination.destination_state_present = true;
    assert!(matches!(
        validate_public_search_controls(&destination),
        Err(SearchControlError::Unsupported(_))
    ));
    let mut non_boolean_path = public_controls(json_tag("int", "1"), json_tag("int", "4"));
    non_boolean_path.return_path = serde_json::from_value(json_tag("int", "1")).unwrap();
    assert_eq!(
        validate_public_search_controls(&non_boolean_path),
        Err(SearchControlError::Invalid("return_path must be bool".into()))
    );
}

fn json_tag(kind: &str, value: &str) -> serde_json::Value {
    serde_json::json!({"kind": kind, "value": value})
}

#[test]
fn native_depth_budget_reserves_the_touch_bfs_suffix() {
    assert_eq!(native_depth_budget(0, 12), (0, 0));
    assert_eq!(native_depth_budget(1, 12), (1, 0));
    assert_eq!(native_depth_budget(4, 12), (1, 3));
    assert_eq!(native_depth_budget(4, 2), (2, 2));
}

#[test]
fn touch_bfs_entry_budget_matches_the_capped_geometric_bound() {
    assert_eq!(touch_bfs_worst_case_entries(3, 2, 13).unwrap(), 13);
    assert!(touch_bfs_worst_case_entries(3, 2, 12).is_err());
    assert_eq!(touch_bfs_worst_case_entries(1, 12, 13).unwrap(), 13);
    assert_eq!(touch_bfs_worst_case_entries(33, 1, 34).unwrap(), 34);
    assert!(touch_bfs_worst_case_entries(255, 12, u64::MAX).is_err());
    assert!(search_budget(2, 1, 33, 1000).is_err());
}

fn graph() -> PermutationGraph {
    PermutationGraph {
        center: vec![0, 1],
        generators: vec![vec![0, 1], vec![1, 0]],
    }
}
fn parse(text: &str) -> beam_core::search::TerminalResult {
    let graph = graph();
    let mut parser = TerminalParser::new(&graph, &[1, 0], Some(3)).unwrap();
    for line in text.lines() {
        parser.line(line).unwrap();
    }
    parser.finish().unwrap()
}
#[test]
fn terminal_replays_original_gather_moves_and_preserves_public_metadata() {
    let result=parse("GLOBAL_BEAM_WIDTH_EFFECTIVE=65\nB_MICRO=32\nSTREAM1_ROWS_PER_JOB=64\nSTREAM3_BATCH_CANDIDATES=64\npuzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 found_depth=1 touch_depth=0 solution=m1\n");
    assert_eq!(result.path, Some(vec![1]));
    assert_eq!(result.effective_beam_width, Some(65));
    assert_eq!(result.metadata["native_seconds"], 0.1);
    assert_eq!(result.metadata["terminal_records"], 1);
    assert_eq!(result.metadata["observed_parent_batch"], 32);
    let result = parse("puzzle_solved=0 puzzle_id=0 seconds=0.2 solution_length=-1 solution=\n");
    assert!(result.path.is_none());
    assert_eq!(result.effective_beam_width, None);
    assert!(result.metadata["observed_parent_batch"].is_null());
}
#[test]
fn malformed_incomplete_contradictory_and_invalid_paths_never_become_not_found() {
    for text in ["startup only","puzzle_solved=1 bad",
        "puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m0",
        "puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=2 solution=m1",
        "puzzle_solved=1 puzzle_id=0 seconds=NaN solution_length=1 solution=m1",
        "puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m3",
        "puzzle_solved=0 puzzle_id=0 seconds=0 solution_length=-1 solution=m1",
        "puzzle_solved=0 puzzle_id=0 seconds=0 solution_length=-1 solution=\npuzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m1",
        "GLOBAL_BEAM_WIDTH_EFFECTIVE=64\nGLOBAL_BEAM_WIDTH_EFFECTIVE=65\npuzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m1",
        "solve_bucket_solution=1 puzzle_id=0 found_depth=1 owner_rank=0 solution_length=1 solution=m1\npuzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m1"] {
        let graph=graph();let mut parser=TerminalParser::new(&graph,&[1,0],Some(3)).unwrap();
        let result=text.lines().try_for_each(|line|parser.line(line)).and_then(|_|parser.finish().map(|_|()));
        assert!(result.is_err(),"accepted {text}");
    }
    let mut bad = graph();
    bad.generators[1] = vec![1, 1];
    assert!(bad.validate().is_err());
    assert!(graph().validate_start(&[0, 0]).is_err());
    let graph = graph();
    let mut parser = TerminalParser::new(&graph, &[1, 0], Some(0)).unwrap();
    assert!(parser
        .line("puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m1")
        .is_err());
}
#[test]
fn collection_is_complete_unique_and_independently_replayed() {
    let text="GLOBAL_BEAM_WIDTH_EFFECTIVE=65\nsolve_bucket_solution=1 puzzle_id=0 found_depth=1 solution_length=1 owner_rank=0 solution=m1\nsolve_bucket_summary=1 puzzle_id=0 records=1 best_length=1\ncollection_status=limit_reached\npuzzle_solved=1 puzzle_id=0 seconds=1 solution_length=1 solution=m1\n";
    let result = parse(text);
    assert_eq!(
        result.metadata["collected_solutions"][0]["path"],
        serde_json::json!([1])
    );
    assert_eq!(result.metadata["collection_status"], "limit_reached");
    // Native summary paths are unquoted and may contain spaces. They are
    // diagnostic output, not fields needed to validate the collection count.
    assert_eq!(
        parse(&text.replace(
            "best_length=1",
            "best_length=1 result_tsv=/tmp/my run/solutions.tsv"
        ))
        .path,
        Some(vec![1])
    );
    for changed in [
        text.replace("records=1", "records=2"),
        text.replace("found_depth=1", "found_depth=2"),
    ] {
        let graph = graph();
        let mut parser = TerminalParser::new(&graph, &[1, 0], Some(3)).unwrap();
        assert!(changed
            .lines()
            .try_for_each(|line| parser.line(line))
            .and_then(|_| parser.finish().map(|_| ()))
            .is_err());
    }
    let budget = search_budget(4, 12, 2, 12345).unwrap();
    assert_eq!(budget.native_forward_depth_limit, 1);
    assert_eq!(budget.effective_touch_bfs_radius, 3);
    assert_eq!(budget.touch_bfs_worst_case_entries, 15);
    assert_eq!(
        search_budget(0, 12, 2, 12345)
            .unwrap()
            .native_forward_depth_limit,
        0
    );
    assert!(search_budget(12, 5, 24, 1_048_576).is_err());
}
