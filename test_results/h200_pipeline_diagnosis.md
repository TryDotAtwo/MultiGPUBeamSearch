# Dual H200 pipeline loss diagnosis — 2026-09-15

## Conclusion and limits

The strongest evidence points to oversized fixed-capacity Stream4 sorting and
associated compute/communication scheduling delays. It is not evidence that
FP16 inference itself suddenly became intrinsically slower. The pipeline both
leaves longer gaps between Stream1/2 graphs and stretches their elapsed spans.
History finalization cannot explain the loss observed in these mid-depth windows.

This is a diagnosis, not a production-profile promotion. No production source or
saved launcher configuration was changed in this diagnostic turn. Nsight was
installed from NVIDIA's configured package repository. Only runtime test settings
were varied. Exact attribution of every extra second in the760.371s full step
would require a full-depth phase trace; the short windows do not provide that.

## Hardware and captures

Vast51125653,2xH200NVL,driver580.178.04,native FP16,original Cube4 puzzle1000.
Binary SHA256 c16711fa988918d0b46f4bb5deb2ee94f0cb848bb3fd327dcdf17268f16e412a.
Nsight Systems2024.6.2.225,trace=cuda,nvtx,sample=none,cpuctxsw=none,
cuda-graph-trace=graph,duration10s,kill=sigterm. Delays110s for100M and180s
for740M. Child processes deliberately ended143 after capture, not solver completion.
The prior unprofiled740M log ends inside depth8 after the interrupted turn:
only depth7=760.371s is a completed large-beam timing. No depth8 result exists.

All three captures cover part of depth7. The740M/32-shard window is an early,
high-pressure part of that depth, not representative of the full-step average.
Analysis trims100ms from each end of the common graph span, yielding9.8044s,
9.8236s,and8.0207s windows. Rates are normalized to these windows.

## Results

Per-GPU averages across the two ranks. GPU graph spans include scheduling and
interleaving; union fractions are not exclusive SM occupancy. Columns overlap
and must not be added into a wall-time budget.

| Requested global beam / shards per rank | Effective beam | Stream4 capacity | Stream4 mean graph ms | Stream4 graph-span union | Stream1/2 graph-span union | NCCL/service stream kernel union | Aggregate sampled parents/s |
|---|---:|---:|---:|---:|---:|---:|---:|
|100M /32|100007936|1562624|9.34|1.82%|93.66%|1.17%|1061165|
|740M /32|740032512|11563008|38.35|32.79%|81.81%|12.00%|656391|
|740M /128|740032512|2890752|18.20|8.09%|90.22%|1.19%|956474|

Mean Stream1/2 graph spans:3.94ms ->4.80ms ->4.00ms, respectively.
The128-shard experiment preserved FP16,micro384,concurrency8,ring12,4 sort slots,
capacity scale1.0,and the exact effective global beam. It reduced each sort
capacity by4x. It is supportive intervention evidence, not a clean identical-
frontier speedup: prefix survivors and thresholds changed (rank0 depth5:
34727462 in the original740M run versus31863946 with128 shards). Frontier identity,
solve quality and full-depth performance must be checked before adopting it.

## Mechanism in source

- cuda/stream4.cu:532 fills the unused sort tail.
- cuda/stream4.cu:536-544 invokes CUB MergeSort with capacity, not scratch_count.
- cuda/stream4.cu:548-559 reduces over capacity again.
- cuda/stream4.cu:303-329 similarly sorts/reduces the histogram at fixed capacity.
- cuda/static_memory.cu:627 sizes fixed CUB temporary storage by shard capacity.

For the original completed depth7,100M/32 launched601 Stream4 jobs on rank0;
740M/32 launched3335. Capacity rises7.3997x and job count5.5491x, giving41.0618x
fixed-capacity item passes before accounting for merge complexity. Per parent,
the item-pass demand is5.5491x larger. This is an algorithmic work proxy, not
measured bandwidth or a claimed41x wall-time slowdown.

Nsight SendRecv mean span grows35.65us ->799.55us in the100M/32 vs740M/32 windows,
with maximum5.34ms ->59.52ms. Count AllReduce mean remains10.47us ->12.43us.
This pattern and the128-shard reduction of communication spans support GPU/peer
readiness delays rather than a general link-latency collapse. Physical NVLink
bandwidth utilization and exclusive kernel resource contention were not measured.

CPU cudaEventSynchronize dominates CPU time in both captures. Correlation to
recorded events places the long waits on the Stream3 completion stream, whose
dependency chain includes upstream inference. It would be incorrect to call
all that CPU wait time exclusive Stream3 work or GPU idle time.

## Next step, not performed

Validate sharding/threshold correctness and identical-frontier semantics, then
measure a complete saturated depth with smaller fixed sort capacities. Tune
shards/trigger/sort concurrency before touching FP16 model accuracy. Removing
sentinel-tail work requires a separately validated graph-compatible design;
no CPU-count hot-path scheduling, hash-table dedup or shard top-k is proposed.

## Evidence

- analyze_dual_h200_graphs.py: read-only reproducible SQLite analysis; stream IDs
  are trace-specific and checked against stream creation order and graph counts.
- h200_graph_profile_analysis.txt and h200_graph_profile_s128.txt: numerical outputs.
- h200_diagnosis_profiles.tar.gz: all three nsys-rep files, capture logs, rank logs,
  previous incomplete740M rank logs, and build/install logs.
- Archive SHA256:5602e45653aba8c4ed343e334590c411dc7f1aaa3a4e55cc095dd962b1bf198e.
- Downloaded archive hash matches the remote file. GPU stop requested after
  preservation; final lifecycle confirmation is recorded separately.
