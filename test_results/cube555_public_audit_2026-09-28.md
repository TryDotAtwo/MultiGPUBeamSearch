# Cube555 public notebook audit

- Public sharing saved; CayleyPy private group has Can edit.
- 68 Python contract/publication tests passed.
- 34 Node schema/replay tests and TypeScript check passed.
- 139 Cloudflare Worker/queue/GitHub writer tests passed.
- Cloudflare deployment f90367b7-fdb9-4b22-ad27-9a52e87f80e5; healthz normal/ok.
- Three complete depth-8 loops on real Kaggle two-T4 hardware; see configs/kaggle_t4_cube555_profiles.json.
- Full initial solve PID1020: native reports length132, 901.376 seconds at beam65536; independent CPU replay against the downloaded competition state passed.
- Downstream results-repo compatibility patch has 40 passing tests. Its external publication awaits separately requested authorization.
- Synthetic fixtures never submitted as competition results.

- Full v1 finished in 2516.856 seconds: PID1020 solved in 132 moves, PID1034 unsolved at depth200. No equal-quality claim against TPU beam16777216.
- Public v2 pushed; all executable remote cells match the committed notebook. Markdown heading has a Windows CLI encoding artifact only. Runtime pin e21b3a61701a0ec21aca80a8b1e1a10dba270610.
- Public sharing and CayleyPy Can edit rechecked after CLI update.
- Real Kaggle v2 PID35 delivery verified in Cloudflare D1: submission 01a0e782-8fb9-7898-942d-f753eea81253, state staged, safe_error null, GitHub staging commit 4cea7132b1cc7cc6ca3fe76b315277ca93505ff7.
- Final-main promotion is not confirmed. Separate results-repo push/PR was rejected by automatic approval review and awaits explicit user authorization.
