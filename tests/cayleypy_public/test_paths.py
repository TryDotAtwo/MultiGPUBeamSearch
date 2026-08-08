import hashlib

import pytest

from tools.cayleypy_public.paths import (
    SolutionRecord,
    apply_path,
    deduplicate_solutions,
    invert_path,
    make_reflected_state,
    tokenize_path,
    validate_original_solution,
)


# The names deliberately do not use a '-' convention: inversion is algebraic.
GENERATORS = {
    "clockwise": (1, 2, 0),
    "counterclockwise": (2, 0, 1),
}


def test_apply_and_invert_round_trip_with_dot_separated_moves():
    initial = (0, 1, 2)
    path = "clockwise.clockwise"

    reached = apply_path(initial, path, GENERATORS)

    assert reached == (2, 0, 1)
    assert invert_path(path, GENERATORS) == "counterclockwise.counterclockwise"
    assert apply_path(reached, invert_path(path, GENERATORS), GENERATORS) == initial


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("", ()),
        ("clockwise", ("clockwise",)),
        ("clockwise.counterclockwise", ("clockwise", "counterclockwise")),
    ],
)
def test_tokenize_path_exposes_strict_dot_separated_tokens(path, expected):
    assert tokenize_path(path) == expected


@pytest.mark.parametrize("path", [None, 1, (), ["clockwise"]])
def test_tokenize_path_rejects_non_string_values(path):
    with pytest.raises(ValueError, match="dot-separated string"):
        tokenize_path(path)


@pytest.mark.parametrize("path", [".", ".clockwise", "clockwise.", "clockwise..counterclockwise"])
def test_tokenize_path_rejects_empty_move_tokens(path):
    with pytest.raises(ValueError, match="empty move tokens"):
        tokenize_path(path)


@pytest.mark.parametrize("path", ["move\nline", "\N{LATIN SMALL LETTER E WITH ACUTE}", "x" * 129])
def test_tokenize_path_rejects_non_printable_non_ascii_or_oversized_tokens(path):
    with pytest.raises(ValueError, match="printable ASCII.*128"):
        tokenize_path(path)


def test_reflection_round_trip_inverts_a_reflected_solution_to_original_solution():
    central = (0, 1, 2)
    initial = (1, 2, 0)
    original_path = "counterclockwise"

    reflected_initial = make_reflected_state(central, original_path, GENERATORS)
    reflected_candidate = "clockwise"
    original_candidate = invert_path(reflected_candidate, GENERATORS)

    assert reflected_initial == (2, 0, 1)
    assert validate_original_solution(reflected_initial, central, reflected_candidate, GENERATORS)
    assert validate_original_solution(initial, central, original_candidate, GENERATORS)


@pytest.mark.parametrize("path", ["unknown", "clockwise..counterclockwise", ".clockwise", "clockwise."])
def test_invalid_or_ambiguous_path_tokens_fail_closed(path):
    with pytest.raises(ValueError):
        apply_path((0, 1, 2), path, GENERATORS)


def test_inversion_requires_a_unique_named_inverse_generator():
    ambiguous = {"a": (1, 2, 0), "b": (2, 0, 1), "also_b": (2, 0, 1)}

    with pytest.raises(ValueError, match="unique inverse.*a"):
        invert_path("a", ambiguous)


def test_solution_record_is_frozen():
    record = _record("original", "counterclockwise", "counterclockwise", (0, 1, 2), 3)

    with pytest.raises(Exception):
        record.path = "clockwise"


def test_solution_record_copies_mutable_reached_state_before_deduplication():
    mutable_state = [0, 1, 2]
    record = _record("original", "counterclockwise", "counterclockwise", mutable_state, 3)

    mutable_state[0] = 9

    assert record.reached_state == (0, 1, 2)
    assert isinstance(record.reached_state, tuple)


def test_validate_original_solution_returns_false_for_non_iterable_generator_value():
    malformed_generators = {"clockwise": None}

    assert validate_original_solution((0, 1, 2), (0, 1, 2), "clockwise", malformed_generators) is False


def test_validate_original_solution_does_not_hide_unrelated_runtime_errors():
    class BrokenMapping(dict):
        def items(self):
            raise RuntimeError("external generator source failed")

    with pytest.raises(RuntimeError, match="external generator source failed"):
        validate_original_solution((0, 1, 2), (0, 1, 2), "", BrokenMapping(clockwise=(1, 2, 0)))


def test_deduplicate_uses_original_path_and_reached_state_then_earliest_provenance():
    first = _record("reflected", "clockwise", "counterclockwise", (0, 1, 2), 8, source="z")
    earlier = _record("original", "counterclockwise", "counterclockwise", (0, 1, 2), 3, source="a")
    distinct_state = _record("original", "counterclockwise", "counterclockwise", (1, 2, 0), 1, source="a")

    records = deduplicate_solutions([first, distinct_state, earlier])

    assert records == [earlier, distinct_state]
    assert hashlib.sha256(earlier.original_oriented_path.encode("utf-8")).hexdigest()


def test_deduplicate_preserves_solver_provenance_on_exact_source_duplicate():
    solver = _record("original", "counterclockwise", "counterclockwise", (0, 1, 2), 1)
    source = _record(
        "source", "counterclockwise", "counterclockwise", (0, 1, 2), 1, source="external-sha"
    )

    assert deduplicate_solutions([solver, source]) == [solver]


def _record(variant, path, original_oriented_path, reached_state, found_depth, source=None):
    return SolutionRecord(
        puzzle_id=7,
        variant=variant,
        path=path,
        original_oriented_path=original_oriented_path,
        found_depth=found_depth,
        touch_depth=0,
        source_solution_sha256=source,
        reflected_source_path=original_oriented_path if variant in {"reflected", "source"} else None,
        valid=True,
        reached_state=reached_state,
    )
