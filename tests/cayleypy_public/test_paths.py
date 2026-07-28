import hashlib

import pytest

from tools.cayleypy_public.paths import (
    SolutionRecord,
    apply_path,
    deduplicate_solutions,
    invert_path,
    make_reflected_state,
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


def test_deduplicate_uses_original_path_and_reached_state_then_earliest_provenance():
    first = _record("reflected", "clockwise", "counterclockwise", (0, 1, 2), 8, source="z")
    earlier = _record("original", "counterclockwise", "counterclockwise", (0, 1, 2), 3, source="a")
    distinct_state = _record("original", "counterclockwise", "counterclockwise", (1, 2, 0), 1, source="a")

    records = deduplicate_solutions([first, distinct_state, earlier])

    assert records == [earlier, distinct_state]
    assert hashlib.sha256(earlier.original_oriented_path.encode("utf-8")).hexdigest()


def _record(variant, path, original_oriented_path, reached_state, found_depth, source=None):
    return SolutionRecord(
        puzzle_id=7,
        variant=variant,
        path=path,
        original_oriented_path=original_oriented_path,
        found_depth=found_depth,
        touch_depth=0,
        source_solution_sha256=source,
        valid=True,
        reached_state=reached_state,
    )
