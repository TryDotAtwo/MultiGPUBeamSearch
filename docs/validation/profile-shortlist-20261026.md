# Service-guided profile shortlist — October26, 2026

The tuner retains its inference-first contract and exact requested beam. A
measured service-pressure heuristic selects at most four complementary component
challengers. Low downstream pressure selects one full-step challenger; other
cases select at most two. If no coarse full-step gain is statistically separated,
refinement stops. Native admission, model validation, complete timing cohorts,
solution replay and incomplete-calibration cache rules remain in force.

Smaller flush thresholds are omitted from the shortlist: they can repeatedly sort
survivor capacity. The trace records analytical per-rank outer jobs, forwards per
model, Stream3 batches and gross sort jobs. These are not measured CUDA kernels
or graph replays. Current release builds disable verbose per-depth job counters.

## Same-host comparison

Quadro RTX5000 16GB, SM75, LibTorch, real Cube555 Transformer+MLP blend .8/.2,
FP16 bodies, existing FP32 readout,30 moves, inference parent batch1024.
Old source7b4f01ff19a5d88328e35e454fbc3b4946fedc56 and the new policy ran
sequentially on the same host. Full fixture widths were exact on every sample.

| Beam | Old tuning | New tuning | Old full step | New full step |
|---:|---:|---:|---:|---:|
|65,536|201.45s|77.82s|5.110s|5.082s|
|262,144|683.87s|285.09s|20.520s|20.689s|

Tuning times cover downstream selection only, excluding native build and
inference calibration. The new runs reused the unchanged inference calibration;
whole cold/warm API durations are not a fair speedup comparison. Each old run
measured20 component and5 full profiles; each new run measured5 and2. Baseline
retained in all four cases. Both widths returned[16,8,1] and independently
replayed to the goal. Both real new-policy cache receipts passed validation.

For new policy,4x width produced4.071x step time:12,895 and12,671 parents/s.
This supports approximately linear scaling over these two saturated widths,
not an extrapolation to all widths, devices, or the VRAM limit. Full-step timing
at the maximum admitted frontier remains unverified; allocation capacity is not
a recommended efficient working maximum. Requested width is never reduced.

Changed-path CPU tests passed, including policy-bound cache validation,
no-gain early stop, frozen inference batch and distinct job-count units.
Retrospective A4000/4060Ti/T4 records informed the heuristic; this new GPU is the
held-out check. There is no global-optimum guarantee.

Evidence archive592 hashed files, SHA256:
`a858c1849f286875d5949bb1835b0e228b7fdcb157a0936f118b1e8050bcd010`.
The owned instance was deleted after archive verification. Approximate compute
and disk cost$0.072, excluding traffic and provider billing rounding.

The separate full clean Kaggle2xT4 run of notebook160/source7b4f01f is documented
in [Cube555 acceptance](cube555-adaptive-20261025.md). It solved index1020 in116
moves with exact beam1,048,576. It predates this heuristic change; no repeat of
that long solve was used for the new policy.
