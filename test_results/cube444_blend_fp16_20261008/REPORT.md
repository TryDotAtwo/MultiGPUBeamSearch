# Native Cube444 FP16 blend — GPU acceptance and matched full-depth comparison

Default execution now converts the checksum-bound original FP32 weights to
FP16 once at load, rejecting nonfinite conversion results. Integer indices
stay integer. Backbone weights/activations and final CUTLASS GEMM operands
are FP16; normalization/softmax reduction and final accumulator, bias/blend
and score quantization are FP32. Explicit FP32 constructor remains an oracle.
Source NPZ identity and 0.6/0.4 coefficients remain unchanged.

Lease54890934, same two physical RTX3060 GPU UUIDs and driver610.43.02 as
the previous full FP32 benchmark. USD0.1888888889/h with40GB disk, remaining
original shared USD10 budget. Deadline watcher session95949 and heartbeat
fp16-blend-2-3060. Evidence and frozen source snapshot are being archived on a separate GitHub branch; no default-branch merge or package release.

Material condition difference: this new lease has a host-set100W limit on
both GPUs, versus170W in the old FP32 run (archived hardware receipt).
Query confirmed active software power caps, no thermal cap. Restoring the
default170W via nvidia-smi was denied Insufficient Permissions; settings did
not change. Within-pair FP16 comparisons share100W, but absolute FP16-vs-old-
FP32 speed ratios confound precision and host power. No clean precision-only
end-to-end speedup is claimed.

Numerical gate on1024 competition states, both GPUs: maximum absolute blend
score error0.0166054, mean0.00385233 against FP32 oracle; per-state argmin
agreement99.21875%, global best256 candidate overlap100%. This corpus gate
does not prove unchanged long-search quality. Fused projection/key parity
against FP32 accumulation of the actual half operands: <=1 integer key unit
on rows1,7,31,64,129,255,256; nonfinite guard accepted. FP16 output types
checked explicitly. Near-goal two-rank three-move solve and independent path
replay passed before launch-order optimization; final order validation pending.

Warmup5, repeats30, batch256 inference measurements (not full depth):

| Operation | GPU0 ms | GPU1 ms |
| --- | ---: | ---: |
| Transformer+readout | 15.8306 | 15.9931 |
| MLP alone | 1.5762 | 1.6630 |
| Transformer+event control | 15.8742 | 16.0351 |
| Readout only | 0.06485 | 0.08738 |
| Readout blend | 0.06586 | 0.08461 |
| Sequential two heads | 17.2776 | 17.4826 |
| MLP-first two-stream blend | 17.2375 | 17.5515 |

The broad comparison is consistent with most extra latency being the MLP
forward (its arithmetic, memory traffic and submission), rather than blend
arithmetic or event overhead. It does not separate MLP FLOPs from its memory
traffic. Small timing differences are not exact causal decompositions.

Old full FP32 pipeline counters independently compared on both ranks:
68224 scorer jobs,34112 Stream3 jobs,119 threshold updates and17465344
final candidates each mode. Stream4 jobs3744 versus3755 (+0.294%); global
request total34930688 each, only rank distribution differs by13241.
History bytes identical. These volume counters support inference-dominated
extra work, but do not measure exact Stream4/cache time contributions.

Old GPU1 scorer calibration at256 parents:25.937600ms only versus29.895008ms
blend. Multiplying the3.957408ms delta by68224 batches predicts269.990s,
close to the actual full-depth276.580s extra. This is a consistency estimate,
not an exact causal percentage: the scorer block includes both model work
and any intra-block memory/resource/submission effects, and clocks can drift.

Profiler trace of original MLP-first order:310 CUDA kernels,2 streams, zero
cross-stream kernel overlap in that recorded pass. MLP kernels0–2.3105ms,
Transformer/readout kernels2.3999–18.7186ms. CPU submission order consumes
the potential overlap window before the long backbone starts.

Controlled alternating-order comparison,30 repeats, batch256, exact key
equality across sequential/MLP-first/Transformer-first paths:
GPU0 17.2673/17.1767/16.9383ms; GPU1 17.3853/17.3936/17.0310ms.
Transformer-first improves over MLP-first by1.39%/2.08%. Trace records
1.1625ms of cross-stream kernel overlap. Production now queues Transformer
before MLP, retaining the shared-parent producer event and final MLP wait.
Profile traces include profiler overhead; speed claims use unprofiled events.

Nsight Compute counter probe failed with ERR_NVGPUCTRPERM. Cache miss rates
and a separate cache-contention contribution are unmeasured; neither zero
cache effects nor a specific cache penalty is established. No host driver or
counter permissions were changed.

Full native FP16 Transformer-only depth PASS1311.46s on both ranks at
34930688 global beam; blend is running on the exact previous losslessly
archived frontier. This is a same-work precision
comparison, not a newly optimized maximum FP16 beam. Existing production4GiB
reserve unchanged; benchmark uses the documented1.25GiB reserve.


## Recovered final results (2026-10-09)

Both full runs exited 0 on both ranks, one AB pair on the identical frontier.
FP16 Transformer 1311.46 s; FP16 blend 1436.21 s; +124.75 s / +9.5123%.
This is one depth at beam 34,930,688, not a solved index1000/256M run.
The short FP32 attribution check also completed PASS; GPU0 Transformer/MLP
medians 28.0226/4.54325 ms, GPU1 30.399/5.11693 ms at batch256.

The source lease54890934 was stopped after its deadline. Restart failed in
Vast OCI runtime. Provider instance-to-instance copy recovered results onto
archive-only helper54975329; no product/GPU test runs on the helper.
No newly recovered result files were downloaded onto Windows.
GitHub connected-app access supports write; HF connected app currently
has read-only scopes. Binary archive remains unverified/unpublished; disks
must not be destroyed until its backup is verified.
