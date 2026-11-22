# Cube555 model artifacts

These classes reproduce the two checkpoint schemas from the public Cube555
input bundle. `models.py` was carried from solver commit
`45a27a93b940e6feb7e61fd13059c7e13997b30f`; the only inference factoring is an
exported `features()` method before each linear Q head.

## Explicit public model objects

```python
from multigpubeamsearch import NativeEnsemble, NativeModel
from multigpubeamsearch.contracts import GraphContract
from tools.cube555.export import export_native_models

contract = GraphContract.from_graph(graph, graph.definition.central_state)
transformer, resmlp = export_native_models(
    MODEL_ROOT / "q555_f1_bell2k.pt",
    MODEL_ROOT / "q555_2k_BEST.pt",
    MODEL_ROOT / "piece_layout_555.json",
    MODEL_ROOT / "puzzle_info.json",
    artifact_dir,                 # new directory; never overwrites old artifacts
    contract=contract,
)
predictor = NativeEnsemble((transformer, resmlp), (TRANSFORMER_WEIGHT, 1.0 - TRANSFORMER_WEIGHT))
```

To reopen a previously exported bundle, use two explicit models and preserve the
same coefficient order:

```python
transformer = NativeModel.for_graph(graph, artifact_dir / "cube555_transformer",
                                   backend="cube555_transformer")
resmlp = NativeModel.for_graph(graph, artifact_dir / "cube555_mlp",
                              backend="cube555_mlp")
predictor = NativeEnsemble((transformer, resmlp), (TRANSFORMER_WEIGHT, 1.0 - TRANSFORMER_WEIGHT))
```

Each coefficient is supplied by the caller. No hidden default blend or implicit
normalization is applied. The resulting predictor goes to the public CayleyPy
native search path; this module does not launch a separate solver or tune batches.

## Boundary and numerical semantics

The graph must have identity center `0..149`, 30 complete gather permutations,
and exactly the checkpoint's ordered move names and permutations. The layout
provides 98 head-facelet tokens plus CLS. Its orbit/local sticker encoding,
piece types and positions are preserved. Sticker IDs use **uint8**, including
128..149. Scripts accept `[B,150]` or `[B,160]`, ignore padding, and return
contiguous FP16 `[B,feature_dim]` tensors.

Each member directory contains:

- `manifest.json`, schema version 1, graph and source SHA256 bindings;
- `features.ts`, whose `forward(states)` returns features before the Q head;
- `output_weight.fp16`, little-endian row-major `[feature_dim,30]`;
- `output_bias.fp32`, little-endian `[30]`, converted from the FP16 source head;
- the exact `layout.json` and `puzzle_info.json` used in export.

Both neural bodies execute in FP16. The existing ensemble readout accumulates
`features.float() @ weight.float() + bias.float()` in FP32, applies the explicit
coefficient and sums members before one clamp/round. This differs numerically
from the legacy `FP16 Linear -> FP32 blend` path. CPU parity reattaches the
original FP16 head to confirm exact feature factoring, then measures this
intentional readout change. GPU ranking/search quality still requires validation.

`hidden1=hidden2=feature_dim`, `residual_count=1`, `normalization="none"` and
`planning_descriptor_only=true` are compatibility fields for the existing
first-member planning parser. They **do not describe the neural body**, which
contains the original LayerNorm/attention/residual operations. They are not
evidence of a memory budget; actual script residency and workspaces must be
accounted for by runtime admission/autotuning.

Artifact validation hashes all five runtime files, rejects duplicate JSON keys,
unsafe/missing file bindings, invalid layout indices, wrong ordered graphs,
nonfinite readout weights and mismatched dimensions. Model snapshots are copied
and reverified through the existing `prepare_model` flow. Binding a graph is a
caller declaration, not proof of historical checkpoint training provenance.

The two new family names require the integrator's matching ensemble feature
loader. Admission tests alone do not establish native/GPU execution support.
