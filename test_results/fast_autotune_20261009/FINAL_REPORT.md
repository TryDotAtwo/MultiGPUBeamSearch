# Exact-frontier autotuning acceptance and small-beam performance audit — 2026-10-09

Tested code: ee50c804defc4f41e5c1151f9c24df46fdf530b1. Native source: d52f81ff7d1b3240996a07fa4ab369b55f84f01f. Runner SHA256: 611ccc84844dde1b0d110bf2015420d1b230340013c2aee6346fecc81dbe1cef.

Workload: LRX13 permutation graph, storage 32 logical length 13, three generated FP16 Hamming MLP heads with coefficients 0.5/0.25/0.25, FP32 accumulation and CUTLASS weighted readout. This is not Cube444 index 1000 or a trained puzzle-solving quality test. All public searches independently replayed a length-4 solution. All full-frontier receipts completed one warmup and five measured complete depths on every GPU; matched inference used the same immutable legal frontier files. Depth medians use the slowest rank in each sample. History/path persistence uses disk, so full-step cost includes the ordinary history output path.

| GPUs | Effective beam | Best bounded Stream1, M parents/s | Matched Stream1, s | Full step, s | Throughput loss | Full calibration + verification, s |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 65,536 | 12.70 | 0.004882 | 0.023199 | 78.95% | 15.26 |
| 2 | 1,048,576 | 12.83 | 0.077445 | 0.181746 | 57.39% | 19.80 |
| 2 | 88,891,392 | 12.87 | 6.528678 | 14.630000 | 55.37% | 256.92 |
| 8 | 65,536 | 48.29 | 0.001318 | 0.036297 | 96.37% | 52.33 |
| 8 | 1,048,576 | 48.82 | 0.019702 | 0.301368 | 93.46% | 50.78 |
| 8 | 327,942,144 | 49.54 | 6.039956 | 104.751000 | 94.23% | 1124.74 |

All final default inference searches selected batch 8192; this is the best found within the default bounded search, not an unbounded optimum. Full verification was enabled explicitly with a 3600-second pipeline budget; maximum-beam totals include large fixture generation and repeated complete depths. Fast default component calibration does not generate these large files. Capacity searches took 1.71 seconds on two GPUs and 8.36 seconds on eight. Maximum width is within the documented shards 1..128 / two staging / one sort-slot policy, not a universal maximum across all allocation policies. The two configurations are different hosts and beam sizes, not a strong-scaling benchmark.

## Small-beam audit requested by Ivan

- The 8x3060 host has two NUMA groups of four GPUs. nvidia-smi reports no supported P2P pairs (CNS within groups, TNS across groups). Raw topology is saved as profile-audit/topology.json.
- At global beam 65,536, each GPU has 8192 parents: one inference microbatch. There is no later inference batch to hide its candidate exchange behind.
- At global beam 1,048,576, the selected baseline has two rings of eight inference slots (8192 parents per slot); all 131,072 local parents are submitted in the initial fill. The existing PREFILL_BEFORE_WAIT switch has no additional parent work to enqueue under that geometry. A different pair of timing medians alone is insufficient to attribute a causal gain to that switch.
- Nsight rank-0 traces of this exact beam show roughly 19 ms of Stream1 kernel work and a diagnostic end-of-depth finalization interval of 135–155 ms after the dispatcher, with substantial NCCL SendRecv and host synchronization. These traces enable PIPELINE_STATS and perturb timing; they are not production performance rows. The profiler-free public full-step receipt is 301.368 ms versus matched inference 19.702 ms.
- Checked staging counts 1/2/4/8, one versus four sort lanes, eight versus four shards, and the existing prefill flag. Instrumented sweep, profiler-free sweep, a closing control, both Nsight reports and SQLite analyses are retained. Profiler-free alternatives measured approximately 301–315 ms, controls approximately 344 ms in the later audit; the original baseline receipt was already 301 ms. No new general optimum or causal prefill speedup is claimed from differing newly initialized communicator contexts. Default scheduling was not changed on that basis.
- Global selection and materialization require the completed depth's candidates. Their tail cannot entirely overlap inference of that same depth. When communication alone exceeds inference, additional rings cannot hide all of it. A heavyweight learned model may have a different limiting stage; this generated Hamming ensemble does not establish that case.

## Verification and boundaries

CPU acceptance: 307 passed, 5 skipped, retained in results-final2 archive external/final-two-setup.log. Native exact A/B union histogram oracle and full rebuild acceptance are retained in results-2gpu. The five-stream architecture, Stream4 CUB selection and semantic global beam contract are preserved. No arbitrary family conversion, 128-GPU hardware acceptance or multi-node launcher is claimed. Source documentation distinguishes exact admission, synthetic service proxies, full-step evidence and maximum-policy scope.

GPU archives are published in this evidence branch and verified by downloading the immutable GitHub multipart archive on the rented host and checking every retained file SHA256. Huge frontier/history binaries are scratch and excluded with sizes/reasons; their legal-generation recipes and hashes are retained. Prepared native runtime artifact-v2 is separately published and SHA256 verified. No GPU archive was copied to Windows. Only owned instances 55029041, 55053060 and 55039547 are cleaned up; unrelated rentals are not touched. Cleanup status is recorded separately.
