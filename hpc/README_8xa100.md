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

# Put start.sh in this directory. The script clones/updates GitHub repo itself.
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

Cleanup/refresh behavior:

- Always removes only `/mnt/pool/6/vokirova/beam8a100/build-a100`.
- Removes only `/mnt/pool/6/vokirova/beam8a100/history` after a successful
  run.
- Keeps `history` after a failed run for diagnostics.
- If `repo/` exists but is not a Git checkout, removes only the exact
  `/mnt/pool/6/vokirova/beam8a100/repo` path before recloning.
- Never removes `repo`, `logs`, `slurm-*.out`, `predict_stats`, or
  `/mnt/pool/3/vokirova/cutlass` during normal job cleanup.
