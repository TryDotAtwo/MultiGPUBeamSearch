import json

from multigpubeamsearch.publication import prepare_publication
import cayleypy as cp
import multigpubeamsearch as beam


class FakePublisher:
    def __init__(self):
        self.payload = None

    def PublisherConfig(self, version, common, contexts, defaults):
        return {"version": version, "common": common, "contexts": contexts, "defaults": defaults}

    def _sha(self, value):
        return "a" * 64

    def expand_solution_row(self, config, row):
        return {"row": row, "context": config["contexts"][row["puzzle_id"]],
                "competition": config["common"]["competition"]}

    def canonical_bytes(self, value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def preflight(self, path):
        self.payload = json.loads(path.read_text(encoding="utf-8"))


def test_publication_uses_caller_graph_and_catalog_identity(tmp_path):
    graph_def = cp.CayleyGraphDef.create(
        [[1, 0]], generator_names=["swap"], central_state=[0, 1]
    )
    graph = cp.CayleyGraph(graph_def, device="cpu")
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text(json.dumps({"cells": []}), encoding="utf-8")
    result = type("Result", (), {
        "solutions": ((0,),),
        "native_metadata": {
            "devices": [0, 1],
            "effective_beam_width": 64,
            "requested_beam_width": 64,
            "profile": {"pipeline_calibration": {"selection": "automatic"},
                        "gpu_names": ["T4", "T4"]},
            "execution_parameters": {
                "B_MICRO": 8, "STREAM1_CONCURRENCY": 1, "STREAM3_RING_SLOTS": 1,
                "SHARD_COUNT": 1, "SHARD_CAPACITY_SCALE_PPM": 1_000_000,
                "STREAM4_BATCH_CANDIDATES": 8, "STREAM4_TRIGGER_CANDIDATES": 8,
                "STREAM4_ACTIVE_SORT_SLOTS": 1,
            },
            "effective_touch_bfs_radius": 0, "solution_mode": "first",
            "requested_max_steps": 2, "max_collected_solutions": 10,
            "model_artifact_hash": "b" * 64, "native_seconds": 0.01,
            "elapsed_seconds": 0.02,
        },
    })()
    model = {"filename": "tiny-model", "sha256": "c" * 64,
             "format": "resmlp-layernorm",
             "manifest": {"state_len": 2, "num_classes": 2, "hd1": 8,
                          "hd2": 8, "nrd": 1, "output_dim": 1,
                          "dtype": "fp16", "normalization": "layernorm",
                          "layout": "test"}}
    publisher = FakePublisher()

    prepare_publication(
        result, graph, [1, 0], puzzle_id=7, predictor=object(), author="alice",
        kaggle_owner="alice", kaggle_slug="generic-search", kaggle_version=3,
        notebook_path=notebook, solver_commit="d" * 40, directory=tmp_path,
        client=publisher, competition="my-graph-set", puzzle_type="my-graph",
        model_metadata=model,
    )

    envelope = publisher.payload["results"][0]
    assert envelope["competition"] == "my-graph-set"
    assert envelope["context"]["puzzle_type"] == "my-graph"
    assert envelope["context"]["generators"] == {"swap": [1, 0]}
    assert envelope["row"]["solution"] == "swap"


def test_publish_result_is_part_of_the_public_api():
    assert callable(beam.publish_result)
