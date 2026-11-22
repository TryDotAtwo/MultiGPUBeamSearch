# Repository consolidation recovery — 2026-11-07

The active branch is main. Historical branches are preserved by `archive/20261107/local/*` tags; source overlays are preserved by the accompanying `archive/20261107/wip/*` tags when the working tree differed from its commit. These tags are recovery snapshots, not accepted runtime releases.

The local recovery directory is `.local/recovery/20261107/`. It contains:

- `complete.bundle`: self-contained Git history, including two independent old clones; SHA-256 `a40fe97f3f3f6333965fb1735c8d3374f01e67b851b1204634edcee22a73050f`.
- `inventory.json`, `checkout-*.json`, and content-addressed `objects/`: original paths, heads, source bytes, raw indexes, staged and unstaged patches, and deletion records.
- `verification.json`: all 24,134 materialized source-file restorations checked against SHA-256.
- `git-verification.json`: independent bundle clone, full fsck and presence of every original checkout commit.
- `moved-checkouts.json` and `moved-build-caches.json`: exact locations of retained original directories. Their Git markers were renamed `.git.archived` so they are archives rather than active checkouts.
- `retained-root-files.json`: model/input/result files restored at their original paths after updating the checkout.

Restore history into a separate directory with `git clone --mirror complete.bundle recovered.git`. A checkout manifest maps every source path to its full SHA-256; the corresponding blob is `objects/<first-two-characters>/<remaining-characters>`. Raw indexes and patches are also addressed by SHA-256. Do not overlay an archive onto a live checkout without comparing its baseline.

The SM120/NVFP4 working source was retained from checkout-18 (`108d0b3a8bf33b78007d1caafc0743d881f34ba5` plus its overlay). It contains experimental kernels and tests; the snapshot does not imply that those variants are production defaults or hardware accepted.

Local models, checkpoints, canonical results and independent third-party/project repositories are not cleanup targets.
