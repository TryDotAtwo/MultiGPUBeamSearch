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
index1020, beam1,048,576, two T4s and140 steps was not executed in the October25
acceptance. The separate October26 clean run is recorded below. Maximum-width
throughput and matched-inference overlap remain unverified.

Evidence archive SHA256 (adaptive run):
`749d2c93a67de1d433fb658a61c801adf70b3515a8a6ab531b2466b05983f84e`.
The155 member hashes were verified before the owned test machine was deleted.

## Clean Kaggle 2xT4 acceptance — October26

Public `trydotatwo/cube555-native-2xt4-blend`, version160, source commit
`7b4f01ff19a5d88328e35e454fbc3b4946fedc56`: all six original code cells were
executed unchanged in a fresh private validation kernel, version3. All cell
SHA256 values were compared with the published source. Kaggle status COMPLETE.

Both real Tesla T4 devices were used, with automatic LibTorch selection and
FP16 Transformer/MLP blend. Exact requested/effective beam1,048,576. Cube555
index1020 returned116 moves before the140-depth limit. Notebook replay and an
independent replay from archived ordered generators and the initial state passed.
The final output cell completed. Total13931.73s includes setup, build,
calibration and search; this is not a per-step or tuning-speed measurement.

Calibration reached its time budget, retained the measured baseline and marked
the bounded search incomplete. The repaired timeout handling therefore passed
the real two-GPU end-to-end case. No globally optimal profile or shortest-path
claim is made. This run predates the service-guided shortlist candidate.

Diagnostics archive159 files, SHA256:
`915dd715b5c2896279a0d6e2a58d76dcaf6cd3497df517470adac05101b956eb`.
