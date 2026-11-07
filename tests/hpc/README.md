# Launcher integration regression (no GPU)

From the repository root, provide a genuine Cube4 data bundle and sealed FP16
weights. The scripts never rent a machine or run CUDA; only the production
runner executable is replaced with a test process. Bundle validation remains real.

```sh
H200_TEST_DATA_DIR=/path/to/data \
H200_TEST_WEIGHT_DIR=/path/to/weights_fp16 \
bash tests/hpc/check_h200_launcher.sh
```

The data directory must contain `puzzle_info.json` and `test.csv` including
puzzle1000. The test verifies eight child environments against the saved profile,
then asserts a missing bundle fails before any rank log is created.
The runner also reports its build metadata: incompatible architecture is rejected
before rank spawn. The fake runner emulates this CLI only; no H200 hardware is
claimed by this test. Binary hashes are recorded, not compared to an approved
release or treated as authentication.
Temporary logs are preserved at the printed paths. This is not numerical GPU acceptance.

On Windows Git Bash, set `H200_PYTHON=python` to use the configured Windows
interpreter. The scripts do not bypass sandbox permission requirements.
