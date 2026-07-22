# IHES Cube Model Run

MEPhI 8xA100 launcher for `cayleypy-ihes-cube` using a real PyTorch model.

The launcher expects the model at:

```text
/mnt/pool/6/vokirova/beam8a100/ihes_cube_model/model.pth
```

Run `prepare_ihes_cube_model.sh` from the visible `basis` shell before `sbatch`.
It downloads the IHES competition data, downloads the model from the GitHub
release asset when missing, and exports the `.pth` model to
`stream1_weights_ihes_bf16`.

Default release model:

```text
p888-t000_1780290207_e40960.pth
```

The prepare script writes `model.pth.release_asset` beside `model.pth`; if the
default release asset changes, the default `model.pth` is re-downloaded and the
Stream1 weights are re-exported. Explicit custom `MODEL_PATH` values keep the
caller-provided file.

The SLURM `start.sh` does not perform network downloads. It only checks prepared
inputs, compiles the runner with IHES state sizing, and runs
`production_runner` through `torchrun` on 8 GPUs.

Default run is intentionally small:

```text
PUZZLE_ID=0
DEPTH_LIMIT=12
BEAM_WIDTH=30000000
```

## Run

From the visible MEPhI terminal on `basis`:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin main --depth 1
git reset --hard origin/main
git log -1 --oneline

cd /mnt/pool/6/vokirova/beam8a100
mkdir -p ihes_cube_model
cp repo/hpc/ihes_cube_model/ihes_cube_model.sh ihes_cube_model/start.sh
cp repo/hpc/ihes_cube_model/prepare_ihes_cube_model.sh ihes_cube_model/prepare.sh
sed -i 's/\r$//' ihes_cube_model/start.sh ihes_cube_model/prepare.sh
bash -n ihes_cube_model/start.sh ihes_cube_model/prepare.sh
chmod +x ihes_cube_model/start.sh ihes_cube_model/prepare.sh

cd ihes_cube_model
./prepare.sh
sbatch -p kaf12 start.sh
```

Override puzzle, depth, or beam width at submit time:

```bash
sbatch -p kaf12 --export=ALL,PUZZLE_ID=0,DEPTH_LIMIT=20,BEAM_WIDTH=48000000 start.sh
```

Logs:

```bash
tail -f /mnt/pool/6/vokirova/beam8a100/ihes_cube_model/logs/ihes-model-<JOBID>.out
```

## Solution Repair CSV

Prepare per-solution repair inputs with one window size, `K1 + 7` by default:

```bash
cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/ihes_cube_model/prepare_solution_repair_single_window.sh ihes_cube_model/
sed -i 's/\r$//' ihes_cube_model/prepare_solution_repair_single_window.sh
chmod +x ihes_cube_model/prepare_solution_repair_single_window.sh
ihes_cube_model/prepare_solution_repair_single_window.sh
```

## Fresh bucket array with public result publishing

For one SLURM array task per puzzle, keep `PUZZLE_LIMIT=1`. `PUZZLE_OFFSET` is the first selected puzzle index; each array task adds `SLURM_ARRAY_TASK_ID * PUZZLE_LIMIT`.

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin main --depth 1
git reset --hard origin/main
git log -1 --oneline

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh ihes_cube_model/
cp repo/hpc/ihes_cube_model/prepare_ihes_prebuilt_runner.sh ihes_cube_model/
sed -i 's/\r$//' ihes_cube_model/ihes_solve_bucket_from_scratch.sh ihes_cube_model/prepare_ihes_prebuilt_runner.sh
bash -n ihes_cube_model/ihes_solve_bucket_from_scratch.sh ihes_cube_model/prepare_ihes_prebuilt_runner.sh
chmod +x ihes_cube_model/ihes_solve_bucket_from_scratch.sh ihes_cube_model/prepare_ihes_prebuilt_runner.sh

PREBUILD_JOB="$(sbatch -p kaf12 --parsable ihes_cube_model/prepare_ihes_prebuilt_runner.sh)"
echo "PREBUILD_JOB=${PREBUILD_JOB}"

unset FRESH_RUN_TAG
export KNOWN_LENGTHS="23 24"
export PUZZLE_OFFSET=2
export PUZZLE_LIMIT=1
export BEAM_PREBUILT_RUNNER="/mnt/pool/6/vokirova/beam8a100/ihes_cube_model/prebuilt-a100-ihes/production_runner"
export PUBLISH_RESULTS_REPO_URL="https://github.com/TryDotAtwo/cayleypy-beam-results.git"
export PUBLISH_RESULTS_DIR="/mnt/pool/6/vokirova/beam8a100/cayleypy-beam-results"

sbatch -p kaf12 --dependency=afterok:${PREBUILD_JOB} --array=0-80%1 --export=ALL ihes_cube_model/ihes_solve_bucket_from_scratch.sh
```

After each puzzle the script writes:

- `data/ihes_cube/puzzles/pXXXX/solutions.tsv`
- `data/ihes_cube/puzzles/pXXXX/unique_solutions.tsv`
- `data/ihes_cube/puzzles/pXXXX/metadata.env`
- `data/ihes_cube/puzzles/pXXXX/summary.md`
- `data/ihes_cube/index.tsv`
- `data/ihes_cube/improvements.tsv`

Publishing is best-effort: local logs remain authoritative if GitHub auth or network is temporarily unavailable.

The array jobs must share BEAM_PREBUILT_RUNNER; otherwise every puzzle task will rebuild the CUDA binary.
