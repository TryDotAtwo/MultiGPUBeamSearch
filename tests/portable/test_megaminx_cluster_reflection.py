from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

from portable.megaminx_cluster.orchestrate import (
    build_reflected_state,
    parse_solution_line,
    build_torchrun_command,
    plan_steps,
    write_synthetic_puzzle,
)
from portable.megaminx_cluster.validate import apply_path, invert_path, validate_solution


GENERATORS = {"a": [1, 2, 0], "-a": [2, 0, 1]}


def test_apply_and_invert_path_roundtrip():
    state = [2, 0, 1]
    path = ["a", "a"]
    reached = apply_path(state, path, GENERATORS)
    assert apply_path(reached, invert_path(path), GENERATORS) == state
    assert invert_path(path) == ["-a", "-a"]


def test_validate_solution_rejects_unknown_or_wrong_path():
    assert validate_solution([2, 0, 1], [0, 1, 2], ["a"], GENERATORS)
    assert not validate_solution([2, 0, 1], [0, 1, 2], ["-a"], GENERATORS)
    with pytest.raises(ValueError, match="unknown move"):
        apply_path([2, 0, 1], ["x"], GENERATORS)


def test_reflected_state_and_candidate_validate_original():
    initial = [2, 0, 1]
    central = [0, 1, 2]
    original = ["a"]
    reflected = build_reflected_state(central, original, GENERATORS)
    reflected_solution = ["-a"]
    candidate = invert_path(reflected_solution)
    assert validate_solution(initial, central, candidate, GENERATORS)
    assert apply_path(reflected, reflected_solution, GENERATORS) == central


@pytest.mark.parametrize(
    ("mode", "has_original", "expected"),
    [("off", False, ("original",)), ("after", False, ("original", "reflected")), ("only", True, ("reflected",))],
)
def test_plans_exact_reflection_steps(mode, has_original, expected):
    supplied = ["a"] if has_original else None
    assert plan_steps(mode, supplied) == expected


def test_reflect_only_requires_supplied_solution():
    with pytest.raises(ValueError, match="original solution"):
        plan_steps("only", None)


def test_parses_rank_solution_line():
    line = "puzzle_solved=1 puzzle_id=900 solution_length=2 solution=a.-a"
    result = parse_solution_line(line, 900)
    assert result == ["a", "-a"]
    with pytest.raises(ValueError, match="no valid solution line"):
        parse_solution_line("puzzle_solved=0", 900)


def test_builds_torchrun_command_with_one_rank_per_gpu(tmp_path):
    command = build_torchrun_command(tmp_path, 4, "job-7-original", 900, 120, 10**9)
    assert command[:2] == ["python3", str(tmp_path / "portable" / "megaminx_cluster" / "torchrun.py")]
    assert "--nproc-per-node=4" in command
    assert "--no-python" in command
    assert command[-3:] == ["900", "120", "1000000000"]


def test_bundled_torchrun_launches_all_static_ranks_without_pytorch(tmp_path):
    launcher = Path("portable/megaminx_cluster/torchrun.py")
    output = tmp_path / "ranks"
    output.mkdir()
    code = (
        "import json,os,pathlib; "
        "p=pathlib.Path(os.environ['RANK_OUTPUT'])/f\"{os.environ['RANK']}.json\"; "
        "p.write_text(json.dumps({k:os.environ[k] for k in "
        "['RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','TORCHELASTIC_RUN_ID']}))"
    )
    env = dict(os.environ, RANK_OUTPUT=str(output))
    result = subprocess.run([
        sys.executable, str(launcher), "--nnodes=1", "--nproc-per-node=4",
        "--node-rank=0", "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0",
        "--rdzv-id=test-static", "--no-python", sys.executable, "-c", code,
    ], env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    rows = [json.loads((output / f"{rank}.json").read_text()) for rank in range(4)]
    assert [row["RANK"] for row in rows] == ["0", "1", "2", "3"]
    assert all(row["WORLD_SIZE"] == "4" and row["LOCAL_WORLD_SIZE"] == "4" for row in rows)
    assert all(row["TORCHELASTIC_RUN_ID"] == "test-static" for row in rows)

def test_writes_run_owned_synthetic_puzzle(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text('initial_state_id,initial_state\n900,"2,0,1"\n')
    target = tmp_path / "run" / "test.csv"
    write_synthetic_puzzle(source, target, 9000900, [1, 2, 0])
    assert source.read_text() == 'initial_state_id,initial_state\n900,"2,0,1"\n'
    assert target.read_text().splitlines()[-1] == '9000900,"1,2,0"'
