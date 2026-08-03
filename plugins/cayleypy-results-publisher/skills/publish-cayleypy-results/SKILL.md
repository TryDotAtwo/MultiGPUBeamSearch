---
name: publish-cayleypy-results
description: Publish complete canonical CayleyPy Kaggle v1 or native v2 result envelopes through the anonymous Cloudflare ingest API and verify receipts without exposing secrets.
---

# Publish CayleyPy results

Use the bundled `scripts/cayleypy_submit.py` for every preflight, submission,
resume, and status check. Do not recreate gzip or HTTP requests manually.

Before submitting, read `../../references/AGENT_PROTOCOL.md` completely. For a
human-facing walkthrough, use `../../references/HUMAN_GUIDE_RU.md`.

## Required behavior

1. Identify whether the source is Kaggle schema v1 or native schema v2.
2. Prefer canonical JSON/JSONL produced by the solver. For a move string or
   CSV/TSV, require a completed `publisher-config.json`.
3. Run `preflight` before any network request.
4. Refuse to invent author, puzzle, proof, reflection, model, hardware,
   profile, Kaggle, SLURM, run, or timing fields.
5. Use the official endpoint pinned in the CLI. Never ask the user for a token
   or endpoint in the normal flow.
6. Preserve the receipt manifest. HTTP 202 means accepted by Cloudflare, not
   yet published to GitHub.
7. Use `--wait` or `poll` when the user asks to verify publication. Claim
   success only for terminal `published` or `duplicate` receipts.
8. Report accepted, published, duplicate, rejected, and unresolved counts plus
   the manifest path. Never print submitted envelopes or secret environment
   values.

Do not modify beam-search or CUDA architecture while using this skill.