# Cube555 outer/model batch correction

User target: outer8192, independent model microbatch, requested beam29,360,128;
reuse existing Transformer pipeline profiles. Python Cube555 regression: 16 passed.
Native C++ launcher previously ignored the separate microbatch. Now it processes
contiguous model chunks and preserves score-ring offsets and stream dependency.
Native Kaggle build, replay smoke and large allocation measurements are pending.

70 Python Cube555/runner unit tests passed with synthetic 128-GiB free-disk metadata (no large allocations); local physical free disk is about8GiB, below the runner default32GiB.
Kaggle capacity audit v2 at81a05ab: exact two T4; isolated8192-parent model sweep selected512 (~1.3303s worst-GPU median vs1.3422s at128). New C++ build and legal four-move native replay smoke passed. Micro64 timing is valid, but its initial quality comparison mistakenly used128; corrected for future runs. Micro512 quality comparison used its actual batch and passed (maxabs0.0500 vs128,128/128best actions). Large capacity checks still running.

Final audit v2 COMPLETE. Allocation probes29,360,128 and33,554,432 passed;58,720,256 failed native required14,874,929,700 > budget14,479,523,840 bytes;67,108,864 also failed. Requested29,360,128 depth5 loop passed: depth4=41.6742s, next local frontier5,648,301 (not saturated14,680,064), sampled peak8475MiB (8.277GiB). History budget allows MAX_DEPTH177 while preserving the requested beam. No complete saturated depth8 or full-puzzle result is claimed for the large width.
