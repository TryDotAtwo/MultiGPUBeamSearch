use beam_core::search::{search_budget, PermutationGraph, TerminalParser};
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
