# IHES Cube Cluster Smoke

Small MEPhI SLURM smoke test for the Kaggle `cayleypy-ihes-cube` format.

This is not a production solve. It checks that the repository can compile and
run with:

- `STATE_LEN=72`
- `STATE_STORAGE_LEN=80`
- `MOVE_COUNT=18`
- generators loaded from `puzzle_info.json`

The script uses dummy 1-output fp16 weights, depth `2`, and beam `512`, so a
non-solved result is expected.

## Run

From the cluster work directory:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin main --depth 1
git reset --hard origin/main
git log -1 --oneline

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/ihes_cube_smoke/ihes_cube_smoke.sh .
sed -i 's/\r$//' ihes_cube_smoke.sh
bash -n ihes_cube_smoke.sh
chmod +x ihes_cube_smoke.sh
sbatch -p kaf12 ihes_cube_smoke.sh
```

To avoid cancelling existing jobs, submit after the newest active `vokirova`
job:

```bash
DEP_AFTER="$(squeue -h -u vokirova -o '%i' | tail -n 1)"
if [ -n "${DEP_AFTER}" ]; then
  sbatch -p kaf12 --dependency=afterany:${DEP_AFTER} ihes_cube_smoke.sh
else
  sbatch -p kaf12 ihes_cube_smoke.sh
fi
```

Logs:

```bash
tail -f /mnt/pool/6/vokirova/beam8a100/ihes_cube_smoke/logs/ihes-smoke-<JOBID>.out
```

Expected final line:

```text
puzzle_solved=0 puzzle_id=0 ... solution_length=-1 solution=
```

The important checks are in the CMake output:

```text
Beam state logical bytes: 72
Beam state physical bytes: 80
Beam move count: 18
```
