# MEPhI 8xA100 Beam Search Launch

Use this only from the visible MEPhI terminal gateway:

```bash
ssh mephi
ssh basis
```

Then prepare and submit:

```bash
JOB=/mnt/pool/6/vokirova/beam8a100
mkdir -p "$JOB"
cd "$JOB"

git clone --branch main --depth 1 https://github.com/TryDotAtwo/MultiGPUBeamSearch.git repo

if [ ! -d /mnt/pool/3/vokirova/cutlass/include ]; then
  mkdir -p /mnt/pool/3/vokirova
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git /mnt/pool/3/vokirova/cutlass
fi

cp repo/hpc/start_8xa100_beamsearch.sh start.sh
sed -i 's/\r$//' start.sh
bash -n start.sh
chmod +x start.sh
sbatch -p kaf12 start.sh
```

To update an existing checkout before resubmitting:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin main --depth 1
git reset --hard origin/main

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/start_8xa100_beamsearch.sh start.sh
sed -i 's/\r$//' start.sh
bash -n start.sh
chmod +x start.sh
sbatch -p kaf12 start.sh
```

Check the job:

```bash
scontrol show job <job_id>
cat /mnt/pool/6/vokirova/beam8a100/slurm-<job_id>.out
tail -n 200 /mnt/pool/6/vokirova/beam8a100/logs/production_runner_p992_d12_b260m_<job_id>.log
```

Run parameters:

- puzzle: `992`
- depth: `12`
- global beam: `260000000`
- GPUs/ranks: `8`
- CUDA architecture: `sm_80`
- solved-neighborhood radius: `5`
- history mode: `static_hybrid`
- Stream3 prediction stats: enabled at `predict_stats_p992_b260m_d12.jsonl`

Cleanup behavior:

- Always removes only `/mnt/pool/6/vokirova/beam8a100/build-a100`.
- Removes only `/mnt/pool/6/vokirova/beam8a100/history` after a successful
  run.
- Keeps `history` after a failed run for diagnostics.
- Never removes `repo`, `logs`, `slurm-*.out`, `predict_stats`, or
  `/mnt/pool/3/vokirova/cutlass` during normal job cleanup.

Network rule:

- `start.sh` does not run `git clone`, `git fetch`, or `git reset` inside the
  compute job. Prepare `repo/` and `/mnt/pool/3/vokirova/cutlass` from the
  visible `basis` shell before `sbatch`.
