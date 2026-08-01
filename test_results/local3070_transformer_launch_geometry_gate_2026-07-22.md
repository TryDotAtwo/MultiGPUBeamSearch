# RTX 3070 transformer launch-geometry gate (2026-07-22)

## Result

Keep the production launch geometry at `B_MICRO=512`, transformer micro-batch 512, concurrency 1, and one ring slot. The `256 x 2` geometry is a real isolated transformer win, but it does not clear the full Stream1->2->3 acceptance gate. No production code or configuration was changed.

## Measurement discipline

- Every accepted batch was submitted through the long-lived Docker GPU queue with its 10-second cooldown.
- `nvidia-smi dmon` showed 0% SM and memory utilization before and after each short batch.
- The earlier game-contaminated series was excluded completely.
- The predeclared acceptance threshold remained at least 3% median improvement with exact outputs.

## Isolated transformer result

Twenty alternating pairs compared the production `512 x 1` launch against `256 x 2` while keeping 512 parents per launch group.

| Geometry | Median, ms/launch group | Result |
|---|---:|---:|
| `512 x 1` | 8.05975 | baseline |
| `256 x 2` | 7.72645 | 4.135% faster |

The candidate won 20/20 pairs. All 20 candidate dumps matched the baseline score payload byte-for-byte. The whole-file SHA differs because the 24-byte dump header records lane geometry; after the header, both layouts have payload SHA-256 `d9566b1856ad86ca7ce9fc258bf5b07f1482ac8df38f3be2b438244986af763a`.

Evidence queue logs: `c68bf1069277`, `c94002e0a55c`, `16385edbfc41`, `d917e9c920d8`, `5a3c0b5ac384`.

## Stream1->2->3 integration gate

For a valid `256 x 2` pipeline configuration, ring slots must also be 2. This preserves the downstream Stream3 batch at 12,288 candidates but doubles Stream1 physical jobs from 8 to 16 for the same frontier.

Across eight alternating integrated pairs:

| Geometry | Median depth-like time, ms | Paired wins | Result |
|---|---:|---:|---:|
| `512 x 1`, ring 1 | 69.0232 | - | baseline |
| `256 x 2`, ring 2 | 68.5261 | 4/8 | 0.720% faster |

The small and inconsistent integrated gain is below the 3% gate. A second `512 x 2`, ring-2 point also regressed median throughput by about 6.2% (`1.320M` versus `1.408M` candidates/s in the same matrix).

Evidence queue logs: `5c89394ac6a8`, `1029d1f61d52`. The failed job `ded3af6e69a0` only established the contract that concurrency cannot exceed ring slots; it is not a performance result.

## Decision

Reject the launch-geometry change for production. It exposes useful overlap in the isolated transformer benchmark, but the extra graph jobs and scheduler work consume the gain in the real pipeline. Retain `512 x 1` until a future scheduler or graph-fusion change can exploit concurrency without doubling physical jobs.