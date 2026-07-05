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
cp repo/hpc/mephi_8xa100_common.sh .
sed -i 's/\r$//' start.sh mephi_8xa100_common.sh
bash -n start.sh mephi_8xa100_common.sh
chmod +x start.sh mephi_8xa100_common.sh
sbatch -p kaf12 start.sh
```

To update an existing checkout before resubmitting:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin main --depth 1
git reset --hard origin/main

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/start_8xa100_beamsearch.sh start.sh
cp repo/hpc/mephi_8xa100_common.sh .
sed -i 's/\r$//' start.sh mephi_8xa100_common.sh
bash -n start.sh mephi_8xa100_common.sh
chmod +x start.sh mephi_8xa100_common.sh
sbatch -p kaf12 start.sh
```

For the 8xA100 tuning workflow, use the three dedicated launchers instead:

```bash
cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/mephi_8xa100_common.sh .
cp repo/hpc/tune_8xa100_stream1.sh .
cp repo/hpc/tune_8xa100_pipeline.sh .
cp repo/hpc/start_8xa100_best.sh .
sed -i 's/\r$//' mephi_8xa100_common.sh tune_8xa100_stream1.sh tune_8xa100_pipeline.sh start_8xa100_best.sh
bash -n mephi_8xa100_common.sh tune_8xa100_stream1.sh tune_8xa100_pipeline.sh start_8xa100_best.sh
chmod +x mephi_8xa100_common.sh tune_8xa100_stream1.sh tune_8xa100_pipeline.sh start_8xa100_best.sh

sbatch -p kaf12 tune_8xa100_stream1.sh
```

After the Stream1 job finishes, launch the production pipeline sweep:

```bash
sbatch -p kaf12 tune_8xa100_pipeline.sh
```

After the pipeline sweep writes `logs/best_pipeline.env`, launch the selected
configuration:

```bash
sbatch -p kaf12 start_8xa100_best.sh
```

The sweep scripts write:

- `logs/best_stream1.env`: best isolated `BEAM_B_MICRO` and
  `BEAM_STREAM1_CONCURRENCY`.
- `logs/tuning_<job_id>/pipeline_sweep_<job_id>.tsv`: per-config pass/fail and
  average full-depth runtime through depth `12`.
- `logs/best_pipeline.env`: fastest passing production config found by the
  pipeline sweep.

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

## Megaminx Transformer Backend Benchmark

Run this before a large 900M+ Megaminx solve on 8xA100 40GB. The default job now benchmarks only Stream1 inference backends and writes the selected backend plus `BEAM_B_MICRO` / `BEAM_STREAM1_CONCURRENCY` to `logs/best_megaminx_transformer_stream1.env`. The rest of the 900M pipeline config stays the known 24-output MLP-style config: 32 shards, Stream3 ring 8, Stream4 batch 262144, Stream4 trigger 1048576, final chunk 98304, final exchange scale 2x, shard capacity scale 1x.

Prepare the repo and scripts on `basis`:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin codex/stream1-piece-transformer --depth 1
git reset --hard FETCH_HEAD
git log -1 --oneline
test -f weights/megaminx_vlad_transformer_fp16/manifest.json

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/mephi_8xa100_common.sh .
cp repo/hpc/bench_8xa100_megaminx_transformer.sh .
cp repo/hpc/start_8xa100_libtorch_megaminx.sh .
sed -i 's/\r$//' mephi_8xa100_common.sh bench_8xa100_megaminx_transformer.sh start_8xa100_libtorch_megaminx.sh
bash -n mephi_8xa100_common.sh bench_8xa100_megaminx_transformer.sh start_8xa100_libtorch_megaminx.sh
chmod +x bench_8xa100_megaminx_transformer.sh start_8xa100_libtorch_megaminx.sh
```

Submit Stream1-only backend/batch/concurrency benchmark:

```bash
cd /mnt/pool/6/vokirova/beam8a100
sbatch -p kaf12 bench_8xa100_megaminx_transformer.sh
```

Default Stream1 benchmark covers the full requested grid for all explicit backend families:

- `B_MICRO` / per-inference batch: `512 1024 2048 4096 8192 12288 16384`;
- concurrency / inference calls per card in one measured group: `1 2 4 8`;
- backend modes: `pytorch:eager`, `libtorch:eager`, `libtorch:cuda_graph`, and `native_cutlass:graph`.

For PyTorch and LibTorch, `B_MICRO` is passed as the benchmark batch size so all rows share the same `batch x concurrency` contract as native CUTLASS. PyTorch remains a reference timing backend and is not selected for production runner output; `best_megaminx_transformer_stream1.env` selects among production-capable `native_cuda_graph` and `libtorch_eager` rows.

To benchmark Stream1 and immediately run one selected 900M depth-8 target pass in the same job:

```bash
cd /mnt/pool/6/vokirova/beam8a100
sbatch -p kaf12 \
  --export=ALL,RUN_SELECTED_900M_AFTER_STREAM1=1,TARGET_DEPTH_LIMIT=8 \
  bench_8xa100_megaminx_transformer.sh
```

The selected 900M pass changes only Stream1 backend, `BEAM_B_MICRO`, and `BEAM_STREAM1_CONCURRENCY`; all other pipeline parameters remain the 24-output 900M config.

Optional Stream1 grid knobs:

```bash
sbatch -p kaf12 \
  --export=ALL,ISOLATED_B_MICRO_SWEEP="512 1024 2048 4096 8192 12288 16384",ISOLATED_CONCURRENCY_SWEEP="1 2 4 8" \
  bench_8xa100_megaminx_transformer.sh
```

Read benchmark results:

```bash
squeue -u vokirova -o "%.18i %.9P %.40j %.8T %.10M %.10L %.20R"
tail -n 200 /mnt/pool/6/vokirova/beam8a100/slurm-<job_id>.out
column -t -s $'\t' /mnt/pool/6/vokirova/beam8a100/logs/tuning_<job_id>/megaminx_transformer_stream1_isolated_<job_id>.tsv
cat /mnt/pool/6/vokirova/beam8a100/logs/best_megaminx_transformer_stream1.env
```

If the selected 900M pass was enabled, also inspect:

```bash
column -t -s $'\t' /mnt/pool/6/vokirova/beam8a100/logs/tuning_<job_id>/megaminx_transformer_bench_<job_id>.tsv
cat /mnt/pool/6/vokirova/beam8a100/logs/best_megaminx_transformer.env
```

To launch a real depth-120 solve with the selected Stream1 config:

```bash
cd /mnt/pool/6/vokirova/beam8a100
source logs/best_megaminx_transformer_stream1.env
export DEPTH_LIMIT=120
export BEAM_WIDTH=900000000
sbatch -p kaf12 --export=ALL start_8xa100_libtorch_megaminx.sh
```
## Megaminx Vlad Transformer LibTorch Run

Use this launcher for the explicit Megaminx `piece_transformer` path. It supports `MEGAMINX_STREAM1_BACKEND=libtorch_eager` and `MEGAMINX_STREAM1_BACKEND=native_cuda_graph`, and keeps the existing Stream2/3/4/5/finalization pipeline.

Prepare the repo and scripts on `basis`:

```bash
cd /mnt/pool/6/vokirova/beam8a100/repo
git fetch origin codex/stream1-piece-transformer --depth 1
git reset --hard FETCH_HEAD
git log -1 --oneline

cd /mnt/pool/6/vokirova/beam8a100
cp repo/hpc/mephi_8xa100_common.sh .
cp repo/hpc/start_8xa100_libtorch_megaminx.sh .
sed -i 's/\r$//' mephi_8xa100_common.sh start_8xa100_libtorch_megaminx.sh
bash -n mephi_8xa100_common.sh start_8xa100_libtorch_megaminx.sh
chmod +x start_8xa100_libtorch_megaminx.sh
```

The exported Vlad transformer weights are tracked in the repo at:

```text
repo/weights/megaminx_vlad_transformer_fp16/manifest.json
```

The launcher uses that directory automatically. You only need `BEAM_WEIGHT_DIR` if you want to override the bundled weights with another exported `piece_transformer` directory.

Submit a first 900M run, defaulting to puzzle `991` and depth `120`:

```bash
cd /mnt/pool/6/vokirova/beam8a100
sbatch -p kaf12 start_8xa100_libtorch_megaminx.sh
```

Useful overrides:

```bash
sbatch -p kaf12 \
  --export=ALL,PUZZLE_ID=991,DEPTH_LIMIT=120,BEAM_WIDTH=900000000 \
  start_8xa100_libtorch_megaminx.sh
```

Default 8xA100-40GB config:

- `BEAM_WIDTH=900000000`
- `SHARD_COUNT=32`
- `SHARD_CAPACITY_SCALE_PPM=1000000`
- `STREAM4_BATCH_CANDIDATES=262144`
- `STREAM4_TRIGGER_CANDIDATES=1048576`
- `BEAM_B_MICRO=8192`
- `BEAM_STREAM1_CONCURRENCY=8`
- `BEAM_STREAM3_RING_SLOTS=8`
- `BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=98304`
- `BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=2000000`

Logs:

```bash
squeue -u vokirova -o "%.18i %.9P %.40j %.8T %.10M %.10L %.20R"
tail -n 200 /mnt/pool/6/vokirova/beam8a100/slurm-<job_id>.out
tail -f /mnt/pool/6/vokirova/beam8a100/logs/production_runner_libtorch_p991_d120_b900000000_<job_id>.log
```

The script writes `nvidia-smi` samples every five seconds under `logs/tuning_<job_id>/` and removes only the job build directory plus successful-run history, using the same guarded cleanup rules as the other MEPhI launchers.
