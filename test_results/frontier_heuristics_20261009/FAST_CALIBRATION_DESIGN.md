# Fast calibration sizing audit,2026-10-09

Status: source-derived equations, executable offline prototype and calculations.
This is NOT an installed fast tuner or a newly GPU-validated performance profile.
No rental was started. The beam's five-stream architecture is unchanged.

## What the current implementation gets wrong for this requirement

1. The native allocation planner has exact byte accounting, but its performance
   score uses arbitrary integer priorities: waves×10^12,jobs×10^9,...,not seconds.
   It does not use this graph/model/cohort's measured stage service rates.
2. Downstream calibration benchmarks at most65536 parents by default and reports
   bounded_legal_frontier for larger beams. That is not proof at the target size.
3. Each inference batch starts one new helper per GPU and constructs/reloads the
   model. Each downstream candidate starts native processes separately for
   requested-beam admission, bounded admission and measurement. Startup work is
   multiplied by candidates even when kernels themselves take milliseconds.
4. Inference cache identity still uses beam_anchor_power instead of exact beam.
   The next implementation must use requested/effective width and actual
   admission context. Never reuse a finalized10M profile for100M or even two
   different requested widths in the same power-of-two bucket.
5. A bigger buffer does not repair a stage whose sustained service rate is below
   the producer. It only postpones backpressure and consumes VRAM.

## Native geometry: actual equations, not independent knobs

Let F be the requested frontier,G the GPU count,S logical shards,a alignment,
U the native config.b_micro value,K move count,r staging slots,A sort workers,
gamma the inherited shard-capacity PPM multiplier,rho the receive multiplier.
For scalar MLP,U includes child rows; it is not the inference parent batch.

```
F_eff = ceil(F/(G*S*a)) * G*S*a
N = ceil(F_eff/G)
L = ceil(N/S)
Q = r*U*K                         # Stream3 transaction capacity
R = ceil(L*r/Q)                  # physical ring_count, approximately L/(U*K)
recv = max(Q,ceil(rho*Q))
C = align(max(ceil(gamma*L),L+max(Q,recv)),a)
```

The same analytical routine must read capacity policies from the native plan,
never silently change them. C is a physical A or B capacity, not a semantic
shard top-k. All final candidates must preserve F_eff as well as F.

Named allocations visible in static_memory.cu:

```
survivor bytes = 64*S*C
histogram bytes = 16*S*307201
Stream1 score/hash arena = 20*R*Q
Stream5 send+receive = 32*network_slots*(Q+recv)
network_slots = R if G>1 else1
Stream4 named sort arrays = 124*A*C
Final A/B union named arrays = 248*C
current frontier = N*state_storage_bytes
```

These are ingredients for a LOWER bound, not the full CUDA memory plan. Prefix
and final layouts overlay; they must be maxed, not added. Persistent survivors
and histograms follow the overlaid prefix. Exact CUB temporary bytes, partition
arrays, final requests/exchange, models/activations, alignment, solver scratch
and driver/runtime reserves remain owned by make_static_memory_plan and the
native all-rank admission. Passing the lower bound NEVER admits a candidate.

Each extra shard alone adds about4.69MiB of histogram storage per GPU. More
shards shrink C and sort arenas, but also increase histogram work and per-shard
graph bookkeeping. The continuous large-shard memory estimate is therefore
`constant + a*S + b/S`, until final-frontier storage dominates the overlay or
the incoming-transaction reserve dominates C. A square-root stationary point
is a seed, not a performance optimum. Enumerate128 integer possibilities with
the actual floor/rounding equations; that is already cheap enough.

Stream4's MergeSort/ReduceByKey uses capacity-sized captured work, including
sentinel tails; final union may sort2*C. Trigger batch J is not the memory
arena C, and lowering J does not automatically make each sort proportional toJ.
Service probes must use the target C and actual sort-item bound. Measure the
merge/hash pass separately from radix score-histogram/partition passes.

## Calculated example: target geometry changes without a transferred profile

Assumptions:8GPU,Cube444 physical state112B,K24,U1024,r2,A1,gamma1.25,rho2,
alignment1024. These are memory-oriented seeds, not measured best profiles.
The full native planner may change/reject a seed after omitted allocations.

| Requested F | Seed F_eff | Seed S | Q | R | C per physical shard | Named memory LOWER bound per GPU |
|---:|---:|---:|---:|---:|---:|---:|
|1,000,000|1,015,808|2|49,152|3|161,792|0.079GiB|
|10,000,000|10,027,008|4|49,152|13|411,648|0.378GiB|
|100,000,000|100,007,936|4|49,152|128|3,906,560|3.558GiB|
|256,000,000|256,016,384|4|49,152|326|10,001,408|9.079GiB|

The10M and100M seed can share S but are different geometries: almost10× the
arena C and R. Therefore identical visible tuning knobs are not evidence that
the same measured profile can be reused. Every exact frontier requires its own
derived memory/latency assessment.

The earlier measured1M Cube profile used gamma4,S4,r8,G2. The prototype
reproduces Q196608,R6,C524288,recv393216 and48physical Stream1 slots exactly.
It does not turn that old measurement into a100M performance result.

The script evaluates2048 geometries in about6ms on this Windows CPU (one run,
not a startup/latency guarantee). It limits memory seed lists to3 candidates
with the same F_eff and nearby memory cost. A measured service-rate benefit may
justify adding a larger-memory candidate; memory alone is not a speed ranking.

## Service-rate heuristics: compute only from measured executor/topology data

Freeze the selected inference parent batch P and measure its time t1 on every
GPU. Use lambda_g=K*P/t1_g as that rank's gross candidate production rate;
do not sum independent rank rates into a global full-step time. Coupled stages
must account for local/remote traffic and the slowest rank/collective.

For a linear transport/partition segment with measured launch latency alpha
and seconds per item beta, queue utilization is `lambda*(alpha/Q+beta)`.
With a chosen utilization ceiling u (prototype default.8, not measured truth):

```
Q >= lambda*alpha / (u-lambda*beta)
```

If the denominator is nonpositive, no amount of buffer growth fixes throughput.
The model must report a bottleneck, not select an enormous ring. MergeSort is
not a linear service curve: probe at the actual target C and retain a measured
size-dependent curve across cache/working-set boundaries.

The pipeline release latency (p90) yields a first staging hint:
`r_hint=max(2,1+ceil(t_release_p90/t_producer_batch))`. Map it back through the
actual R/Q/C equations and exact native memory plan. More slots are useful only
when observed waits/starvation require them; extra slots can raise C through
the writable reserve. Queue maxima and acceptance/nonfinite flags remain gates.

Stream4 arrival/reprocessing factor is graph/model/depth dependent. The current
native fixed flow_scale2.5 is an assumption, not a measured discard probability.
Estimate it from counters and a bounded state-score sample; preserve uncertainty
and probe adverse occupancy. For concurrent workers use measured aggregate
throughput under shared memory bandwidth, never divide time by worker count.
A logarithmic arrival envelope can seed receive reserves, but may not replace
the communicator's protocol reserve or exact worst-case admission contract.

## Implementation order to make user calibration fast

1. Batch the exact native plan queries in one process/context per GPU. Exchange
   all-rank admission results once; reject incompatible beam/cohort/byte plans.
2. Keep a calibration helper alive per GPU: load models once, send candidate
   batch commands, reset timing/memory peaks and return correctness/telemetry
   evidence. Keep selected executor and FP16/FP32 semantics identical to search.
3. Fit short executor/service curves and compute geometry for the target F.
   Use analytical pruning to obtain2–3 downstream candidates instead of30+.
   Do not rebuild/export models for each candidate. Baseline inference cache
   may provide a prior curve, never a target-frontier validated profile.
4. Probe only the component workloads needed to distinguish those candidates:
   actual target-size sort/union arenas, transaction partition/communication,
   and a short overlapped producer/consumer window with exact target allocations.
   If confidence intervals overlap, keep the admitted conservative candidate;
   do not spend minutes proving a tiny gain. Mark this component_calibrated.
5. Validate with the first actual working depth at the target frontier. This is
   real search progress, not six additional artificial huge-depth repetitions.
   Do not resize buffers inside a running depth. Bad observed throughput or
   changed flow means unverified/invalidated evidence, not silent verified-cache
   acceptance; any new geometry takes effect at an explicitly safe launch
   boundary. Before the beam fills, that observation is not full-size evidence.
6. Cache a finalized profile only under exact F/F_eff, graph/model/build/executor,
   GPU UUID/topology/world and memory-context identities. Keep inferred,
   component-calibrated and full-target-depth verified statuses distinct.

A sub-second CPU decision is plausible and the arithmetic is measured. A
5–15second GPU calibration target on small/medium geometry is only a hypothesis
until the persistent helper/component probes are implemented and GPU-tested.
For a frontier whose one actual sort/step is slow, physics imposes that cost.
We must not promise instant fully validated256M profiles.

## Verification / remaining work

Prototype tests check native1M geometry, exact-F_eff comparators, shard histogram
cost, ring-reserve coupling, lower-bound rejection, integer limits and linear
service stability. They do not prove GPU speed or exact total memory admission.
The prototype is isolated under test_results and has not changed library/native
behavior. Persistent calibration, batched native plan queries, service telemetry
integration and exact-beam cache keys remain implementation work.
