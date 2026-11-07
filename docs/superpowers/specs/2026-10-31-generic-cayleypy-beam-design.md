# Generic CayleyPy beam notebook

## Intent

Keep `multigpubeamsearch` as the native multi-GPU engine and keep calling
CayleyPy's `graph.beam_search`. Remove the public notebook's dependency on the
Cube555 convenience module so a user can swap in a different CayleyPy graph,
start state set, and predictor without rewriting search, tuning, reflection,
result validation, or publication flow.

## Approach

Keep puzzle/model construction at the notebook's problem-definition boundary;
move reusable path conversion, replay validation, and result writing into the
generic `multigpubeamsearch` API. Keep publication metadata explicit and
parameterized by the problem and model. Do not add a new wrapper around the
CayleyPy search call or move GPU execution into CayleyPy.

The notebook's editable problem definition supplies `graph`, `start_states`,
and `predictor`, plus optional reflection paths and result identity metadata.
All search controls remain available in the existing configuration cell.

## Constraints and limits

- Preserve the current Cube555 blend as the runnable default example.
- Keep collect, BFS touch, reflection, result replay/save, publication, profile
  reuse, and automatic device/profile selection available.
- A graph must satisfy the native permutation-graph contract.
- Native inference accepts registered model exporters only. A new neural
  architecture needs its own exporter/backend registration; it must not be
  silently substituted with another model.
- Keep the visible notebook readable; no embedded hashes, audit machinery, or
  generated profile tables.
