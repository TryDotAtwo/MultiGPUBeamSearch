# Two-H200 native FP16 large-beam campaign — 2026-09-15

Status: 100M completed; 740M run in progress. This is a bounded performance
experiment on original Cube4 puzzle1000, not a claimed solved puzzle.

## Identity and reproducibility

- Vast51125653,2xH200NVL,143771MiB reported per GPU, NV6 interconnect.
- Driver580.178.04,CUDA12.8.93,NCCL2.28.9; CUTLASSv3.9.2.
- Rental$8.000/hr GPU +$1.112/hr800GB disk =$9.112/hr; network$40/TB in/out.
- Repository TryDotAtwo/MultiGPUBeamSearch,branch codex/hopper-stream1-fusion.
- Initial source3341c89a; history registration fix ab1f248a, DMA fix be4f06c4.
- FP16 manifest03f55e5b617874cbbfaace5796f7e348588eb2c0b5ef079e45c06b05d84d0cb0.
- puzzle_info2ae03f88108d9818315a1b36abf383bfab450bc6c1aeb6c0b6a2e502c8f05f48.
- test.csv e9a734df2a63656477c2c6f9920f490188fdfa8a780027f1b91783e82b77419e.

## Isolated full Stream1

Both GPUs tested concurrently; seven alternating same-binary pairs each,
micro384,concurrency8,300 CUDA Graph replays. No FP8.

| GPU | Original FF2 median parents/s | SM90 FP16 FF2 median parents/s |
|---|---:|---:|
|0|549035.3|581102.8|
|1|549295.7|581534.8|

All36 paired score files (18 per GPU:11 CSV-seeded plus7 synthetic) byte-identical.
This is limited numerical evidence, not population solve-quality validation.
Ignore the report's legacy full-network FLOP count for CLS-reduced inference.

## Whole pipeline

| Hardware/profile | Effective global beam | Depth7 s | Depth8 s | Depth8 parents/s | Rental/hr |
|---|---:|---:|---:|---:|---:|
|1xH200SXM,previous FP16 optimized|100007936|165.351|165.364|604774.5|$5.789|
|2xH200NVL,FP16|100007936|92.3284|92.2702|1083859.5|$9.112|
|2xH200NVL,large FP16|740032512|pending|pending|pending|$9.112|

100M bounded run exit0 on both ranks,unsolved; total255.228s on rank0.
Dual100M achieves93.22% of the sum of isolated medians. Compared with the previous
SXM result this is1.792x throughput, not a controlled same-hardware scaling test:
SXM/NVL,driver,and host differ.

Shared knobs: outer parent batch384,model micro384,concurrency8,ring12,
32 logical shards per rank,two resident shard buffers,4 active sort slots,
capacity scale1.0,alignment1024,final chunk65536,exchange scale2.0.
No semantic shard top-k or cap was introduced.

## Capacity and history

750M rejected by exact budget gate:145439819088 required bytes vs145005281280
budget bytes, retaining4294967296 bytes runtime headroom per GPU.
740M per-rank static allocation142399013376 bytes; nonstatic estimate1115620176.
Phase1 scratch100957151488;phase2 scratch26664297728;phase3 scratch77113907968.
These phases alias the same scratch pool: do not sum them.
Observed running memory138577MiB/GPU,4580MiB free. Peak verification pending.

History mode remains static_hybrid, global128GiB RAM +600GiB disk.
Per-rank three pinned slots total35521560576bytes, staging50331648bytes,
compressed RAM arena33147584512bytes,disk arena322122547200bytes.
Worst-case60 full740032512-state history layers require710431211520bytes
(16bytes/entry), versus710540263424bytes of compressed arenas: only109051904bytes
nominal margin. Real initial layers are much smaller; nevertheless a60-depth
production run should have additional history reserve and has not been executed.

Large single cudaHostAlloc failed on this host even with idle GPUs:
1.60GB passed;4/8/10GiB failed with cudaErrorInvalidValue.
A contiguous11.84GB allocation registered in1GiB ranges passed.
The first integration exposed D2H transfers spanning independent registrations;
final history copies now use matching1GiB pieces on the existing stream, with
one completion event after all copies. No allocation was added to the depth loop.
Linux registration opt-in is enabled only in the dual-H200 launcher; default
allocation elsewhere remains unchanged. Extended smoke crosses1GiB and passed.

## Cost interpretation

At100M,60*depth8 is92.27minutes,$14.01. Using measured first9 depths plus51
steady depths gives82.68minutes,$12.56. Excludes startup/compilation,transfer,
restarts and reflections. A puzzle of solution length60 is not guaranteed to
be solved by60 search depths. Large-beam estimate pending actual filled-beam timing.

## Evidence and cleanup

dual_initial_results.tar.gz SHA256:
4e9d758fdcaf5ba54f0de6b0abc44e4fe8ac38de047e4c18a01c1d5e92df200e.
Contains both isolated GPU score/log sets,100M rank logs,and initial capacity failures.
Final large-beam archive and instance deletion verification pending.
