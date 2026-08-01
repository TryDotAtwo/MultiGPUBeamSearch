from __future__ import annotations

from dataclasses import dataclass

from portable.megaminx_cluster.autotune.controller import (
    BudgetController,
    ControllerConfig,
    TrialRequest,
    run_session,
)
from portable.megaminx_cluster.autotune.evidence import EvidenceStore, SessionIdentity
from portable.megaminx_cluster.autotune.probe import ProbeResult


RUNTIME = {
    "b_micro": 8192, "stream1_concurrency": 1, "stream3_ring_slots": 1,
    "shard_count": 8, "shard_capacity_scale_ppm": 1250000,
    "stream4_batch_candidates": 262144, "stream4_trigger_candidates": 524288,
    "stream4_active_sort_slots": 1, "final_materialize_chunk_candidates": 65536,
}


class Clock:
    def __init__(self): self.now = 0.0
    def __call__(self): return self.now


def identity(budget=21600):
    return SessionIdentity(
        "A100", 40960, 80, 8, "570", "abc", "model", "release", "mlp",
        "output_move_count", (900, 950, 1000), 30_000_000, budget,
        5, 24, 16, 256 * 2**20, 8_308_825,
    )


def test_budget_reserves_final_repetitions_before_screening():
    clock = Clock()
    budget = BudgetController(clock, total_seconds=100, initial_estimate_seconds=10)
    assert budget.can_launch(reserve_jobs=8)
    clock.now = 11
    budget.observe(11)
    assert not budget.can_launch(reserve_jobs=8)


def test_run_session_orders_phases_and_derives_bfs_only_once(tmp_path):
    clock = Clock()
    calls: list[TrialRequest] = []

    def probe(trial):
        calls.append(trial)
        clock.now += 1
        stable = trial.beam <= 60_000_000
        metrics = {"wall_us": 1_000_000, "peak_vram_mib": [35000] * 8}
        return ProbeResult(stable, "stable" if stable else "oom", metrics, ())

    store = EvidenceStore.create_or_resume(tmp_path, identity())
    result = run_session(ControllerConfig(identity(), RUNTIME, 120_000_000), probe, clock, store)
    phases = [call.phase for call in calls]
    assert phases[0] == "max_beam"
    assert "halving" in phases
    assert phases[-1] == "final"
    assert result.maximum_stable_beam == 60_000_000
    assert result.bfs_radius == 5
    assert all(call.bfs_radius == 5 for call in calls)
    assert (tmp_path / "resume.json").is_file()
    assert (tmp_path / "registry.fragment.json").is_file()


def test_hard_deadline_emits_unverified_partial_session(tmp_path):
    clock = Clock()

    def slow_probe(trial):
        clock.now += 20
        return ProbeResult(True, "stable", {"wall_us": 20_000_000, "peak_vram_mib": [1] * 8}, ())

    ident = identity(budget=45)
    store = EvidenceStore.create_or_resume(tmp_path, ident)
    result = run_session(ControllerConfig(ident, RUNTIME, 120_000_000), slow_probe, clock, store)
    assert result.complete is False
    fragment = result.registry_fragment
    assert all(
        anchor["status"] == "unverified"
        for anchor in fragment["profiles"][0]["anchors"].values()
    )


def test_resume_skips_completed_trial_keys(tmp_path):
    clock = Clock()
    store = EvidenceStore.create_or_resume(tmp_path, identity())
    store.write_checkpoint({"completed_keys": ["max_beam:30000000:seed:900:0"]})
    store.append_trial({"key": "max_beam:30000000:seed:900:0", "phase": "max_beam", "beam": 30000000, "stable": True, "status": "stable", "wall_us": 1000000, "config_id": "seed", "puzzle_id": 900, "repetition": 0})
    seen = []

    def probe(trial):
        seen.append(trial.key)
        clock.now += 1
        stable = trial.beam <= 30_000_000
        return ProbeResult(stable, "stable" if stable else "oom", {"wall_us": 1_000_000, "peak_vram_mib": [1] * 8}, ())

    run_session(ControllerConfig(identity(), RUNTIME, 60_000_000), probe, clock, store)
    assert "max_beam:30000000:seed:900:0" not in seen

def test_successive_halving_selects_fastest_survivor(tmp_path):
    clock = Clock()

    def probe(trial):
        clock.now += 0.01
        wall = 1_000_000 // int(trial.runtime["stream1_concurrency"])
        stable = trial.beam <= 30_000_000
        return ProbeResult(stable, "stable" if stable else "oom", {"wall_us": wall, "peak_vram_mib": [100] * 8}, ())

    store = EvidenceStore.create_or_resume(tmp_path, identity())
    result = run_session(ControllerConfig(identity(), RUNTIME, 60_000_000), probe, clock, store)
    anchors = result.registry_fragment["profiles"][0]["anchors"]
    assert next(iter(anchors.values()))["runtime"]["stream1_concurrency"] == 2
