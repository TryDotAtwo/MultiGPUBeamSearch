# Codex work history export verification

Date: 2026-08-03
Branch: `agent/complete-project-work-history`
Source selector: session metadata `cwd == D:\100XH100`

## Coverage

- Source JSONL logs: 339
- Unique session ids: 300
- Source bytes represented: 1,721,547,729
- Exported session files: 339
- User entries: 16,863
- Codex entries: 41,648
- Tool calls: 47,318
- Tool outputs: 47,352
- Exported session bytes: 394,546,546
- Largest exported file: 59,391,065 bytes (below GitHub's 100 MiB hard file limit)
- Manifest entries: 339
- Missing manifest targets: 0
- Duplicate archive paths: 0

## Public-safety scan

The exporter redacted 49 GitHub-token-shaped values, 5 Hugging Face token values, 92 private-key blocks, 1,483 opaque encrypted inter-agent payloads, and 330 generic credential assignments. A post-export filename-only scan found zero GitHub, Hugging Face, AWS, GitLab, Slack, private-key, and unredacted Bearer-token signatures. Remaining long opaque-looking matches were inspected and were embedded ZIP/image base64 payloads, not authentication material.

## Requirement trace

The historical C++/CUDA-only constraint is present in the May sessions, including the `no_pytorch`, `C++CUDA_only` contract and the request to implement Stream 1 with CUTLASS/custom inference rather than PyTorch runtime kernels.

## Structural checks

- `python -m py_compile tools/export_codex_history.py`: PASS
- Manifest target existence/uniqueness: PASS
- Session file count equals source log count: PASS (339/339)
- `git diff --cached --check -- ':'!history/codex'`: PASS for authored/exporter/index metadata. Historical transcript bodies intentionally preserve original terminal trailing whitespace.
