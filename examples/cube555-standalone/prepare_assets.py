"""Prepare the public Cube555 models and source dependencies on an online host."""

import shutil
import sys
from pathlib import Path

from multigpubeamsearch import download_bundle, load_graph, setup_sources
from multigpubeamsearch.contracts import GraphContract
from tools.cube555.export import export_native_models


def main(assets, repository, cache):
    bundle_url = (
        "https://github.com/TryDotAtwo/MultiGPUBeamSearch/releases/download/"
        "cube555-inputs-20261001/cube555-inputs.zip"
    )
    assets.mkdir(parents=True)
    inputs = download_bundle(bundle_url, cache_dir=cache / "inputs")
    sources = setup_sources(source_dir=repository, cache_dir=cache / "sources")
    shutil.copytree(sources.cutlass_dir, assets / "cutlass")

    model = inputs / "model"
    graph_info = model / "puzzle_info.json"
    graph = load_graph(graph_info)
    contract = GraphContract.from_graph(graph, graph.definition.central_state)
    export_native_models(
        model / "q555_f1_bell2k.pt",
        model / "q555_2k_BEST.pt",
        model / "piece_layout_555.json",
        graph_info,
        assets / "models",
        contract=contract,
    )
    shutil.copy2(graph_info, assets / "puzzle_info.json")
    shutil.copy2(inputs / "competition" / "test.csv", assets / "test.csv")


if __name__ == "__main__":
    main(*(Path(value).resolve() for value in sys.argv[1:]))
