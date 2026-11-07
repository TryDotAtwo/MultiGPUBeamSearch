import json
from pathlib import Path


SOURCE = Path("test_results/kaggle_source_models2/cayleypy-rw-models2-tetraminx.ipynb")
OUT = Path("kaggle_tetraminx_multigpu")

CELL = r"""%%time
import os, re, shutil, subprocess, time
from pathlib import Path

MG_REPO = Path('/tmp/mg_beam_repo')
MG_BUILD = Path('/tmp/mg_beam_build')
MG_CUTLASS = Path('/tmp/cutlass')
MG_WEIGHTS = Path('/tmp/tetraminx_stream1_weights')
MG_OUT = Path('/kaggle/working/multigpu_beam')
for p in (MG_REPO, MG_BUILD, MG_WEIGHTS):
    if p.exists():
        shutil.rmtree(p)
MG_OUT.mkdir(parents=True, exist_ok=True)

def checked(cmd, cwd=None, env=None):
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)

checked(['git', 'clone', '--depth', '1',
         'https://github.com/TryDotAtwo/MultiGPUBeamSearch.git', MG_REPO])
if not (MG_CUTLASS / 'include').exists():
    if MG_CUTLASS.exists():
        shutil.rmtree(MG_CUTLASS)
    checked(['git', 'clone', '--depth', '1',
             'https://github.com/NVIDIA/cutlass.git', MG_CUTLASS])

inference_model = unwrap_model(get_inference_model(cfg, model, averaged_model))
checkpoint = MG_OUT / 'tetraminx_output1.pth'
torch.save({'model': inference_model.state_dict()}, checkpoint)
checked([
    'python3', MG_REPO / 'tools/export_stream1_mlp.py',
    '--weights', checkpoint, '--out', MG_WEIGHTS,
    '--format', 'batchnorm-folded', '--dtype', 'fp16',
    '--num-classes', str(cfg['num_classes']),
], cwd=MG_REPO)

puzzle_info = INPUTS_DIR / 'puzzle_info.json'
test_csv = INPUTS_DIR / 'test.csv'
checked([
    'cmake', '-S', MG_REPO, '-B', MG_BUILD, '-GNinja',
    '-DCMAKE_BUILD_TYPE=Release', '-DBEAM_CUDA_ARCHITECTURES=75',
    f'-DCUTLASS_DIR={MG_CUTLASS}',
    f'-DBEAM_PUZZLE_INFO_JSON={puzzle_info}',
    '-DBEAM_ENABLE_DEBUG=ON', '-DBEAM_ENABLE_DEPTH_LOGS=ON',
])
checked(['cmake', '--build', MG_BUILD, '--target', 'production_runner', '-j', '2'])

world_size = 2
global_beam = int(cfg['multigpu_beam_width'])
shards = 8
alignment = 1024
effective_beam = ((global_beam + world_size * shards * alignment - 1)
                  // (world_size * shards * alignment)
                  * (world_size * shards * alignment))
local_beam = effective_beam // world_size
shard_capacity = (((local_beam + shards - 1) // shards * 1050000 + 999999)
                  // 1000000)
shard_capacity = ((shard_capacity + alignment - 1) // alignment) * alignment
stream3_batch_candidates = 49152 * 4
shard_capacity = max(shard_capacity, stream3_batch_candidates)
print(f'requested_beam={global_beam} effective_beam={effective_beam} '
      f'local_beam={local_beam} shard_capacity={shard_capacity} '
      f'stream3_batch_candidates={stream3_batch_candidates}')

env0 = os.environ.copy()
env0.update({
    'BEAM_WEIGHT_DIR': str(MG_WEIGHTS),
    'BEAM_GENERATOR_PATH': str(puzzle_info),
    'BEAM_PUZZLE_INFO_JSON': str(puzzle_info),
    'BEAM_TEST_CSV': str(test_csv),
    'BEAM_RUNTIME_CONFIG_MODE': 'manual',
    'BEAM_B_MICRO': '49152',
    'BEAM_STREAM1_CONCURRENCY': '4',
    'BEAM_STREAM3_RING_SLOTS': '4',
    'BEAM_SHARD_COUNT': str(shards),
    'BEAM_SHARD_BUFFER_COUNT': '2',
    'BEAM_SHARD_CAPACITY_CANDIDATES': str(shard_capacity),
    'BEAM_STREAM4_BATCH_CANDIDATES': '98304',
    'BEAM_STREAM4_TRIGGER_CANDIDATES': '98304',
    'BEAM_STREAM4_ACTIVE_SORT_SLOTS': '4',
    'BEAM_GLOBAL_SPILL_CAPACITY': '0',
    'BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM': '1000000',
    'BEAM_GPU_HEADROOM_BYTES': str(224 * 1024**2),
    'BEAM_HISTORY_MODE': 'ram',
    'BEAM_HISTORY_SLOT_COUNT': '2',
    'BEAM_HISTORY_WORKERS': '1',
    'BEAM_HISTORY_RAM_BYTES': str(28 * 1024**3),
    'BEAM_HISTORY_DISK_BYTES': str(32 * 1024**3),
    'BEAM_SOLVED_NEIGHBORHOOD_RADIUS': str(cfg.get('hashed_neigbourhood', 0)),
    'BEAM_DEPTH_LOG_EVERY': '1',
})
solved_re = re.compile(
    r'puzzle_solved=(\d+) puzzle_id=(\d+) seconds=([0-9.eE+-]+) '
    r'solution_length=(-?\d+) solution=(.*)$')
list_states_to_solve = list(cfg['list_states_to_solve']) or list(range(len(df_test)))
final_paths, list_solutions_lenghts, list_incorrect_paths, rows = [], [], [], []

for puzzle_id in list_states_to_solve:
    history = Path('/tmp/beam_history_arena') / f'tetraminx_{puzzle_id}'
    if history.exists():
        shutil.rmtree(history)
    env = env0.copy()
    env['BEAM_HISTORY_DISK_PATH'] = str(history)
    nccl_file = Path('/tmp') / f'tetraminx_nccl_{puzzle_id}.bin'
    if nccl_file.exists():
        nccl_file.unlink()
    env['BEAM_NCCL_ID_FILE'] = str(nccl_file)
    cmd = [
        'python3', '-m', 'torch.distributed.run', '--no-python',
        '--nnodes=1', '--nproc-per-node=2', '--node-rank=0',
        '--rdzv-backend=c10d', '--rdzv-endpoint=127.0.0.1:29500',
        f'--rdzv-id=tetraminx_{puzzle_id}',
        MG_BUILD / 'production_runner', str(puzzle_id),
        str(cfg['max_steps']), str(global_beam),
    ]
    started = time.perf_counter()
    proc = subprocess.run(list(map(str, cmd)), cwd=MG_REPO, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    log = MG_OUT / f'tetraminx_{puzzle_id}.log'
    log.write_text(proc.stdout, encoding='utf-8')
    print(f'multigpu puzzle={puzzle_id} rc={proc.returncode} seconds={elapsed:.3f}')
    if proc.returncode:
        print(proc.stdout[-8000:])
        raise RuntimeError(f'MultiGPUBeamSearch failed for puzzle {puzzle_id}')
    matches = [solved_re.search(line) for line in proc.stdout.splitlines()]
    matches = [m for m in matches if m]
    solved_matches = [m for m in matches if int(m.group(1))]
    match = solved_matches[0] if solved_matches else (matches[0] if matches else None)
    solved = bool(match and int(match.group(1)))
    solution = match.group(5) if solved else ''
    length = int(match.group(4)) if solved else None
    sample = str(df_sample_submission['path'].iloc[puzzle_id])
    final_paths.append(solution if solved and length < len(sample.split('.')) else sample)
    list_solutions_lenghts.append(length if solved else np.nan)
    rows.append({'puzzle_id': puzzle_id, 'solved': int(solved),
                 'solution_length': length, 'seconds': elapsed,
                 'return_code': proc.returncode, 'log': str(log)})
    if history.exists():
        shutil.rmtree(history)

multigpu_results = pd.DataFrame(rows)
multigpu_results.to_csv(MG_OUT / 'tetraminx_multigpu_results.csv', index=False)
display(multigpu_results)

# Submission is produced inside the tested integration cell so no optional
# legacy reporting or analysis cell can affect kernel completion.
new_paths = []
solved_by_id = dict(zip(list_states_to_solve, final_paths))
for puzzle_id in range(len(df_sample_submission)):
    new_paths.append(str(solved_by_id.get(puzzle_id, df_sample_submission['path'].iloc[puzzle_id])))
df_submission = df_sample_submission.copy()
df_submission['path'] = new_paths
df_submission.to_csv('/kaggle/working/submission.csv')
print('submission_path=/kaggle/working/submission.csv rows=', len(df_submission), flush=True)
"""


def main():
    nb = json.loads(SOURCE.read_text(encoding="utf-8"))
    nb["cells"][6]["source"] += (
        "\n# MultiGPUBeamSearch uses a global beam across two T4 ranks.\n"
        "cfg['multigpu_beam_width'] = 2**21\n"
    )
    # Remove a stray non-Python token preserved in the upstream %%time model-size cell.
    for cell in nb["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        for stray_token in ("\n\u0438\u0437\n", "\n\u00e8\u00e7\n"):
            source = source.replace(stray_token, "\n")
        cell["source"] = source
    nb["cells"][51]["source"] = CELL
    nb["cells"][51]["metadata"] = {}
    for cell in nb["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        if "paper_list = [paper_dict[key] for key in paper_keys]" in source:
            cell["source"] = (
                "# The original paper-table reporting cell is intentionally skipped.\n"
                "# It is unrelated to training, solving, or submission generation and\n"
                "# assumes optional metadata keys that are absent in this run.\n"
                "print('Skipping optional paper-table report')\n"
            )
            cell["metadata"] = {}
    # Submission is the terminal artifact; drop optional legacy analysis cells after it.
    nb["cells"] = nb["cells"][:52]
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
    OUT.mkdir(parents=True, exist_ok=True)
    notebook_name = "cayleypy-rw-models2-tetraminx-multigpu.ipynb"
    (OUT / notebook_name).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "trydotatwo/cayleypy-tetraminx-multigpu-beam",
        "title": "CayleyPy Tetraminx MultiGPU Beam",
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": ["cayley-py-professor-tetraminx-solve-optimally"],
        "kernel_sources": [],
        "model_sources": [],
    }
    (OUT / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
