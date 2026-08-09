# Project Agent Rules

## Operating Rule
- Read `AGENTS.md` before project edits.
- Update `memory/CHANGELOG.md` after meaningful code, config, or architecture changes.
- Update `memory/PROMPTS.md` with user prompts that define requirements.
- Store test outputs, logs, and verification notes under `test_results/`.
- Prefer agent-centered code: explicit contracts, deterministic tests, small files, low hidden state.

## Branch Model
- `main` is the protected release and integration branch for the whole repository.
- `stream1` is the long-lived integration branch for Stream 1 model architecture support.
- MLP, Transformer, shared-contract, and future Stream 1 work uses short-lived branches from `stream1` and merges back only after cross-architecture verification.
- Cluster/HPC, Kaggle notebooks, and result publishing remain separate verification domains and must not be silently bundled into a Stream 1 change.
- Canonical branch names describe the subsystem and task; do not use an agent name such as `codex/` as the branch namespace.

## AI Development Attribution
- OpenAI Codex running GPT-5.6 Sol is the primary implementation agent for architecture, code, tests, verification, and technical documentation in this repository.
- Ivan Litvak owns project direction, infrastructure decisions, review, and acceptance.
- Every commit substantially implemented by Codex must end with this exact trailer:
  `Co-authored-by: OpenAI Codex (GPT-5.6 Sol) <codex@openai.com>`
- Pull requests must identify the Codex model used, the human direction, executed verification commands, hardware used, and any verification that remains unavailable.
- Do not rewrite historic commits solely to add attribution; document historic collaboration in `CONTRIBUTORS.md`.

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
