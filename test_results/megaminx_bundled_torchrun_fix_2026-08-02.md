# Bundled torchrun clean-cluster fix

- Date: 2026-08-02
- Cluster evidence: job 33285 failed in 4 seconds before GPU work.
- Root cause: `/usr/bin/python3` could not import `torch.distributed.run` on a clean node.
- Architecture: bundled single-node static torchrun-compatible launcher; no PyTorch wheel, compiler, network, or system package required.
- RED: command-contract and real four-rank launcher tests both failed against the old archive.
- GREEN: focused tests reached 2 passed; full portable suite reached 185 passed, 3 skipped, with one stale fixture failure; after adding the required launcher to that fixture, the failing test passed.
- Local Windows note: pytest reported completion but did not exit before the command timeout; direct two-rank launcher smoke also passed.
- Required live gate: rebuild SM80 archive, download cleanly, and rerun through SLURM on 8x A100.
