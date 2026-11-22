# Project Agent Rules

## Operating Rule
- Read `AGENTS.md` before project edits.
- Update `memory/CHANGELOG.md` after meaningful code, config, or architecture changes.
- Update `memory/PROMPTS.md` with user prompts that define requirements.
- Store test outputs and full logs in remote `test_results/` and GitHub Actions artifacts; keep only small source-controlled verification notes locally.
- Prefer agent-centered code: explicit contracts, deterministic tests, small files, low hidden state.

## Architecture Source
- Primary architecture contract: `ARCHITECTURE_NEED.md`.
- Current Stream 4 implementation rule: use threshold + compact + CUB/fixed-temp sort/reduce + compact; do not use Stream 4 hash-table dedup.

## Implementation Guardrails
- `State128` logical bytes are `v[0..119]`.
- `State128` padding bytes are `v[120..127]`.
- Persistent frontier padding must be zero.
- `FinalResponse` may temporarily store `target_local_idx` in `v[120..123]`.
- `Hash128` is one logical 128-bit key stored as `{lo, hi}`.
- `CandidateMeta` must remain 32 bytes and 32-byte aligned.
- Stream 3 payload id must be original candidate id, not compact index.
- Stream 4 must not apply shard top-k or semantic shard cap.

## Local code only; execution through GitHub (2026-11-17)

- This user instruction overrides older local-download and local-results guidance.
- Windows may hold source code and small source-controlled project instructions. Do not download or create experiment datasets, model weights, checkpoints, result archives, runtime/build dependencies, binaries, or full execution logs on the user's computer, including both C: and D:.
- Commit/publish code through GitHub. Dispatch remote work using GitHub Actions on GitHub-hosted Linux runners. Computation and experiment files remain on the project's remote compute/storage; publish reports and artifacts to GitHub without downloading them to Windows.
- Do not run ssh/scp/sftp, local backup watchers, local compute, or self-hosted runners on the user's Windows computer. SSH/SCP between a GitHub-hosted runner and the owned remote instance is permitted.
- Use background API/connector/gh calls for control and summaries. Do not open terminals, browsers, or other windows. If an authorized Windows helper is essential, use CREATE_NO_WINDOW / -WindowStyle Hidden and never give it a data-download path.
- Respect the project's current resource ownership, budget, and pause state. A transport smoke check does not authorize renting or starting paid compute.
- Preserve existing unique data and evidence. This policy is not permission for bulk deletion. Do not restart disabled legacy download/watch scripts.

