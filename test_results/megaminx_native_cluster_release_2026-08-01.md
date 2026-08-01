# Megaminx native cluster release verification вЂ” 2026-08-01

Branch: `codex/megaminx-native-cluster-release`

## Implemented contract

- One-puzzle `run.sh` в†’ `sbatch` в†’ `torchrun`, one process per selected GPU.
- Mandatory GPU list, beam, and puzzle; missing puzzle never invokes `sbatch`.
- Reflection modes `off`, `after`, and `only`, with independent CPU replay.
- Exact hardware, VRAM class, SM, world size, backend/model class, and beam
  power profile matching. No runtime fallback is present.
- Separate deterministic archive names for sm75/sm80/sm86/sm89/sm90/sm120.
- Archive inspection requires exactly one native SM and rejects PTX/JIT.
- Cloudflare Worker schema v2 publication, durable 202 receipt persistence,
  safe retry classification, no redirects/private endpoints, status polling,
  and reflected-source provenance.
- Public payload scans reject secret-like material and private absolute paths.
  The packaged model manifest intentionally omits the source checkpoint path.

## Verification executed locally

```text
py -m pytest tests/portable -q
109 passed, 2 skipped in 2.04s
```

The two skips are platform-specific: Bash execution and symlink creation are
not available in this Windows test environment. Static Bash contract tests,
Python command construction tests, deterministic tar.zst tests, fake
`cuobjdump` gates, v2 publisher HTTP tests, replay tests, metadata tests, and
workflow matrix tests passed.

`git diff --check` passed with no whitespace errors.


## Independent review fixes

A read-only code review identified and the branch fixed packaged-root resolution,
SLURM CUDA visibility preservation, importable archive layout, independent archive
allowlist/hash/private-content verification, host-only glibc/loader/driver policy,
public publish-only retry mode, and mandatory README/preflight wrapper staging.

## Worker v2 external evidence

The companion Worker branch `codex/cayleypy-results-ingest` at `ce40a15`
reported 72 schema tests and 138 Worker/Queue/GitHub tests passing. Its staging
deployment accepted a live H100x4 SM90 v2 fixture with HTTP 202 and published
submission `019fbdc0-9347-7639-895c-d27e703694ad` under `data/v2/slurm/...`.
Production was not changed.

## Hardware readiness boundary

The build matrix and packaging gates cover all six requested SM targets, but
this Windows machine cannot compile or execute CUDA/Linux artifacts. No native
archive produced here is claimed as hardware-smoked.

The committed registry currently contains measured Kaggle 2xT4 beam profiles.
A100x7, H100x4, L4, RTX 30/40, and RTX PRO 6000 Blackwell profiles remain
deliberately non-runnable until repeated exactness-gated sweeps on those exact
family/VRAM/world-size tuples are imported with
`tools/measure_megaminx_cluster_profiles.py`. This is a release gate, not a
fallback: those configurations fail before solving instead of borrowing T4 or
another GPU's settings.

The GitHub workflow publishes only a prerelease and requires a self-hosted
Linux CUDA 12.8 build runner. Production release promotion should occur only
after `cuobjdump`, archive content, and solve/replay smokes on each claimed
hardware tuple.
