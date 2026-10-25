# Cube555 and adaptive profile calibration — 2026-10-25

The native adapter accepts the ordered 150-sticker Cube555 graph and explicit
Transformer/ResMLP blend. Logical beam width is exact; shard alignment only
changes physical buffers. Inference microbatch is selected first and frozen.

Downstream candidate generation uses GPU SM/L2 properties, native VRAM budget,
GPU count, graph/state dimensions, exact beam and measured service times.
At most16 coarse component candidates and two bounded refinement rounds are
considered. Full steps decide the winner; component estimates only shortlist.
Cache reuse requires matching identities/search trace and fresh native admission.
This bounded measured search does not certify a global optimum.

## Verified workloads

- RTX A4000 16GB: exact beams65,536/131,072; full steps5.76661/11.1153s with
  the preceding seven-profile policy. Ordinary16-depth search preserved65,536
  for12 saturated depths. Short solutions independently replayed at both widths.
- RTX4060Ti16GB: new adaptive policy, real Cube555 Transformer+MLP(.8/.2),
  FP16 bodies with existing FP32 readout; width65,536, inference batch128.
  17 component profiles,5 full profiles, one refinement round. Baseline retained
  at5.656s/step; no statistically separated2% improvement among neighbors.
- Fresh calibration283.43s, cached calibration2.670s; both inference and
  pipeline caches hit on a fresh process. Short scramble[0,9,17] returned
  [16,8,1], independently replayed on both calls.
- Exact-width CPU arithmetic includes uneven rank quotas and world sizes up to128.
  This is not GPU128 acceptance. The latest targeted Python set passed18 tests;
  preceding geometry/component checks also passed.

Maximum28,474,368 on the A4000 was admitted and allocated with one-parent model
execution, not timed as a full maximum frontier. The notebook default competition
index1020, beam1,048,576, two T4s and140 steps has not been executed in this
acceptance. No full default solve, multi-GPU Cube555 throughput or matched-inference
overlap claim is made. A notebook publication is not runtime acceptance.

Evidence archive SHA256 (adaptive run):
`749d2c93a67de1d433fb658a61c801adf70b3515a8a6ab531b2466b05983f84e`.
The155 member hashes were verified before the owned test machine was deleted.
