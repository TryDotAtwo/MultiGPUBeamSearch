# Private 2xT4 cube4 ReLU Transformer smoke solve
GITHUB_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
GITHUB_BRANCH = "codex/stream1-piece-transformer"
EXPECTED_COMMIT_PREFIX = "2ff65a0"
KAGGLE_MODEL_SOURCE = "trydotatwo/cube4-full-transformer-inference"
MODEL_SOURCE_SLUG = "cube4-full-transformer-inference"
MODEL_INPUT_ROOTS = ["/tmp/cube4_bundle"]
WEIGHT_OUT_DIR = "/kaggle/working/stream1_transformer_weights_fp16"

BEAM_WIDTH = 65_536
START_PUZZLE_ID = 0
PUZZLE_COUNT = 1
DEPTH_LIMIT = 100
RUN_TIMEOUT_SEC = 0
CUDA_ARCHITECTURES = "75"

TORCHRUN_NNODES = 1
TORCHRUN_NPROC_PER_NODE = 2
TORCHRUN_NODE_RANK = 0
TORCHRUN_RDZV_BACKEND = "c10d"
TORCHRUN_RDZV_ENDPOINT = "127.0.0.1:29500"
TORCHRUN_RDZV_ID = "cube4_transformer_libtorch_smoke"

ENABLE_DEBUG = False
ENABLE_DEPTH_LOGS = True
ENABLE_DEBUG_LOGS = False
DEBUG_STREAM_TIMING = False
DEBUG_INFERENCE_TRACE = False
DEBUG_PATH_TRACE = False
DEBUG_FINAL_VALIDATE = False
DEBUG_FINAL_EXCHANGE_TRACE = False
DEBUG_FINAL_HISTOGRAM_TRACE = False
DEBUG_STREAM4_HISTOGRAM_TRACE = False
DEBUG_DEPTH_FLOW_TRACE = False
DEBUG_PIPELINE_STATS = False

RUNTIME_CONFIG_MODE = "manual"
SHARD_BUFFER_COUNT = 2
STREAM4_BATCH_ALIGNMENT = 1024
SHARD_CAPACITY_SCALE_PPM = 1_250_000
GLOBAL_SPILL_CAPACITY = 0
STREAM5_RECV_CAPACITY_SCALE_PPM = 1000000
GPU_HEADROOM_BYTES = 256 * 1024**2
B_MICRO = 384
STREAM1_CONCURRENCY = 1
STREAM3_RING_SLOTS = 2
SHARD_COUNT = 2
STREAM4_ACTIVE_SORT_SLOTS = 2
STREAM4_BATCH_CANDIDATES = 16_384
STREAM4_TRIGGER_CANDIDATES = 16_384

DEPTH_LOG_EVERY = 1
PUZZLE_LOG_EVERY = 1
HISTORY_MODE = "static_hybrid"
HISTORY_SLOT_COUNT = 2
HISTORY_WORKERS = 1
HISTORY_RAM_BYTES = 2 * 1024**3
HISTORY_DISK_BYTES = 2 * 1024**3
HISTORY_DISK_PATH = "/tmp/beam_history_transformer_native_10m_p990"
SOLVED_NEIGHBORHOOD_RADIUS = 0
SOLVED_NEIGHBORHOOD_MAX_ENTRIES = 0
STREAM2_SUFFIX_RADIUS = 0
STREAM2_SUFFIX_BACKEND = "composed_permutations"
STREAM2_SUFFIX_MAX_COUNT = 0
STOP_ON_FAILURE = True
LIVE_LOG_RANKS = [0]


from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

WORK_DIR = Path('/kaggle/working')
TMP_DIR = Path('/tmp')
REPO_DIR = TMP_DIR / 'beam_solver_cube4_libtorch'
CUTLASS_DIR = TMP_DIR / 'cutlass'
BUILD_DIR = TMP_DIR / 'beam_build_cube4_libtorch'
WEIGHT_OUT_DIR = Path(WEIGHT_OUT_DIR)
RUN_LOG_DIR = WORK_DIR / 'cube4_transformer_libtorch_logs'
RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)


def run_checked(cmd, cwd=None, env=None):
    cmd = [str(part) for part in cmd]
    print('+ ' + ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_capture(cmd, cwd=None, env=None, check=True):
    cmd = [str(part) for part in cmd]
    print('+ ' + ' '.join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end='', flush=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout)
    return result


def round_up(value, alignment):
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def torchrun_world_size():
    world_size = int(TORCHRUN_NNODES) * int(TORCHRUN_NPROC_PER_NODE)
    if world_size <= 0:
        raise ValueError(f'invalid torchrun topology: nnodes={TORCHRUN_NNODES} nproc={TORCHRUN_NPROC_PER_NODE}')
    return world_size


def derived_values():
    world_size = torchrun_world_size()
    beam_alignment = world_size * int(SHARD_COUNT) * int(STREAM4_BATCH_ALIGNMENT)
    global_beam_effective = round_up(BEAM_WIDTH, beam_alignment)
    local_beam_width = global_beam_effective // world_size
    logical_shard_size = (local_beam_width + int(SHARD_COUNT) - 1) // int(SHARD_COUNT)
    shard_capacity = round_up((logical_shard_size * int(SHARD_CAPACITY_SCALE_PPM) + 999999) // 1000000, STREAM4_BATCH_ALIGNMENT)
    stream3_batch = int(STREAM3_RING_SLOTS) * int(B_MICRO) * 24
    return {
        'world_size': world_size,
        'beam_alignment': beam_alignment,
        'global_beam_effective': global_beam_effective,
        'local_beam_width': local_beam_width,
        'logical_shard_size': logical_shard_size,
        'shard_capacity': shard_capacity,
        'stream3_batch': stream3_batch,
    }


def disk_line(path):
    usage = shutil.disk_usage(path)
    return f'{path}: free={usage.free} total={usage.total}'


def preflight():
    values = derived_values()
    history_required_bytes = int(values['global_beam_effective']) * int(DEPTH_LIMIT) * 16
    history_budget_bytes = int(HISTORY_RAM_BYTES) + int(HISTORY_DISK_BYTES)
    available_ram_bytes = int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
    free_tmp_bytes = int(shutil.disk_usage('/tmp').free)
    print('history_preflight=', {
        'entry_bytes': 16,
        'required_bytes_at_depth_limit': history_required_bytes,
        'ram_budget_bytes': int(HISTORY_RAM_BYTES),
        'disk_budget_bytes': int(HISTORY_DISK_BYTES),
        'total_budget_bytes': history_budget_bytes,
        'available_ram_bytes': available_ram_bytes,
        'free_tmp_bytes': free_tmp_bytes,
    }, flush=True)
    if HISTORY_MODE == 'static-hybrid':
        if history_budget_bytes < history_required_bytes:
            raise RuntimeError(f'history budget {history_budget_bytes} is below required {history_required_bytes}')
        if int(HISTORY_RAM_BYTES) + 4 * 1024**3 > available_ram_bytes:
            raise RuntimeError('history RAM budget leaves less than 4 GiB host headroom')
        if int(HISTORY_DISK_BYTES) + 4 * 1024**3 > free_tmp_bytes:
            raise RuntimeError('history disk budget leaves less than 4 GiB /tmp headroom')
    print('KAGGLE_MODEL_SOURCE=', KAGGLE_MODEL_SOURCE, flush=True)
    print('GITHUB_BRANCH=', GITHUB_BRANCH, flush=True)
    print('torchrun_topology=', {
        'nnodes': TORCHRUN_NNODES,
        'nproc_per_node': TORCHRUN_NPROC_PER_NODE,
        'node_rank': TORCHRUN_NODE_RANK,
        'world_size': values['world_size'],
        'rdzv_endpoint': TORCHRUN_RDZV_ENDPOINT,
    }, flush=True)
    print('derived_config=', values, flush=True)
    print('disk_tmp=', disk_line('/tmp'), flush=True)
    print('disk_working=', disk_line('/kaggle/working'), flush=True)
    run_capture(['nvidia-smi'], check=False)
    import torch
    gpu_count = torch.cuda.device_count()
    print('torch_cuda_device_count=', gpu_count, flush=True)
    if gpu_count < int(TORCHRUN_NPROC_PER_NODE):
        raise RuntimeError(f'2xT4 solve requires at least {TORCHRUN_NPROC_PER_NODE} CUDA devices, found {gpu_count}')
    if int(STREAM1_CONCURRENCY) > int(STREAM3_RING_SLOTS):
        raise RuntimeError('STREAM1_CONCURRENCY must be <= STREAM3_RING_SLOTS')
    if values['stream3_batch'] > values['shard_capacity']:
        raise RuntimeError(f'stream3_batch={values["stream3_batch"]} exceeds shard_capacity={values["shard_capacity"]}')
    if int(STREAM4_BATCH_CANDIDATES) > values['shard_capacity']:
        raise RuntimeError(f'STREAM4_BATCH_CANDIDATES={STREAM4_BATCH_CANDIDATES} exceeds shard_capacity={values["shard_capacity"]}')
    if int(STREAM4_TRIGGER_CANDIDATES) > values['shard_capacity']:
        raise RuntimeError(f'STREAM4_TRIGGER_CANDIDATES={STREAM4_TRIGGER_CANDIDATES} exceeds shard_capacity={values["shard_capacity"]}')
    return values


def cleanup_path(path):
    path = Path(path)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def find_model_checkpoint():
    input_root = Path('/kaggle/input')
    all_pth = sorted(input_root.rglob('*.pth')) if input_root.exists() else []
    candidate_roots = [Path(path) for path in MODEL_INPUT_ROOTS if Path(path).exists()]
    if not candidate_roots and input_root.exists():
        candidate_roots = sorted({path for path in input_root.rglob('*') if path.is_dir() and MODEL_SOURCE_SLUG in str(path)})
    matches = []
    for root in candidate_roots:
        matches.extend(root.rglob('*.pth'))
    matches = sorted(set(matches))
    if len(matches) != 1:
        message = [
            'expected exactly one transformer .pth under the configured Kaggle model source',
            f'configured_model_source={KAGGLE_MODEL_SOURCE}',
            'candidate_roots=' + json.dumps([str(path) for path in candidate_roots], indent=2),
            'matched_pth=' + json.dumps([str(path) for path in matches], indent=2),
            'all_discovered_pth=' + json.dumps([str(path) for path in all_pth], indent=2),
        ]
        raise RuntimeError('\n'.join(message))
    return matches[0]


# Validate the private cube4 input dataset and build deterministic CayleyPy-format data.
import hashlib
import csv
import torch

model_candidates = sorted(Path('/kaggle/input').rglob('model.pth'))
if len(model_candidates) != 1:
    raise RuntimeError(f'expected exactly one model.pth under /kaggle/input, got {model_candidates}')
BUNDLE_DIR = model_candidates[0].parent.parent
INPUT_DATASET = BUNDLE_DIR
print('resolved_bundle_dir=', BUNDLE_DIR, flush=True)

MODEL_PATH = BUNDLE_DIR / 'model' / 'model.pth'
METADATA_PATH = BUNDLE_DIR / 'model' / 'model.json'
GENERATOR_PATH = BUNDLE_DIR / 'generators' / 'p002.json'
TARGET_PATH = BUNDLE_DIR / 'targets' / 'p002-t000.pt'
EXPECTED_SHA256 = {
    MODEL_PATH: '58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164',
    METADATA_PATH: '87a6d644400b5f5e6c5258e7bd9562427488163d5b7542f73c86aa1948b5f17e',
}
for path, expected in EXPECTED_SHA256.items():
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    print('sha256=', path.name, actual, flush=True)
    if actual != expected:
        raise RuntimeError(f'SHA256 mismatch for {path}: {actual} != {expected}')

spec = json.loads(GENERATOR_PATH.read_text(encoding='utf-8'))
target = torch.load(TARGET_PATH, map_location='cpu', weights_only=True).to(torch.uint8).tolist()
move_names = list(spec['move_names'])
moves = list(spec['moves'])
if len(target) != 96 or len(move_names) != 24 or len(moves) != 24:
    raise RuntimeError('unexpected cube4 generator/target contract')
# Twelve-move deterministic scramble, avoiding immediate inverse pairs.
scramble_indices = [0, 2, 4, 8, 10, 12, 16, 18, 20, 6, 14, 22]
initial = list(target)
for move_index in scramble_indices:
    initial = [initial[source] for source in moves[move_index]]
print('known_scramble=', [move_names[i] for i in scramble_indices], flush=True)

CUBE4_DATA_DIR = WORK_DIR / 'cube4_data'
cleanup_path(CUBE4_DATA_DIR)
CUBE4_DATA_DIR.mkdir(parents=True)
(CUBE4_DATA_DIR / 'puzzle_info.json').write_text(json.dumps({
    'central_state': target,
    'generators': {name: move for name, move in zip(move_names, moves)},
}, indent=2) + '\n', encoding='utf-8')
with (CUBE4_DATA_DIR / 'test.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['initial_state_id', 'initial_state'])
    writer.writerow([0, ','.join(map(str, initial))])

preflight_values = preflight()
for path in (REPO_DIR, BUILD_DIR, WEIGHT_OUT_DIR):
    cleanup_path(path)

run_checked(['git', 'clone', '--branch', GITHUB_BRANCH, '--depth', '1', GITHUB_REPO_URL, REPO_DIR])
checked_out_commit = run_capture(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO_DIR).stdout.strip()
if not checked_out_commit.startswith(EXPECTED_COMMIT_PREFIX):
    raise RuntimeError(f'expected commit {EXPECTED_COMMIT_PREFIX}, got {checked_out_commit}')
print('checked_out_commit=', checked_out_commit, flush=True)
checkpoint_path = find_model_checkpoint()
print('selected_transformer_checkpoint=', checkpoint_path, flush=True)
run_checked([
    sys.executable, REPO_DIR / 'tools' / 'export_stream1.py',
    '--weights', checkpoint_path,
    '--out', WEIGHT_OUT_DIR,
    '--format', 'piece-transformer',
    '--dtype', 'fp16',
    '--num-classes', '6',
    '--metadata', METADATA_PATH,
    '--generators', GENERATOR_PATH,
    '--source-root', BUNDLE_DIR,
], cwd=REPO_DIR)
manifest = json.loads((WEIGHT_OUT_DIR / 'manifest.json').read_text(encoding='utf-8'))
if manifest.get('backend') != 'piece_transformer':
    raise RuntimeError(f'exported manifest backend is not piece_transformer: {manifest.get("backend")!r}')
print('exported_manifest_summary=', {
    'backend': manifest.get('backend'),
    'dtype': manifest.get('dtype'),
    'seq_len': manifest.get('seq_len'),
    'd_model': manifest.get('d_model'),
    'layers': manifest.get('num_layers'),
    'output_dim': manifest.get('output_dim'),
}, flush=True)

if not (CUTLASS_DIR / 'include').exists():
    cleanup_path(CUTLASS_DIR)
    run_checked(['git', 'clone', '--depth', '1', 'https://github.com/NVIDIA/cutlass.git', CUTLASS_DIR])

run_checked([
    'cmake', '-S', REPO_DIR, '-B', BUILD_DIR, '-GNinja',
    '-DCMAKE_BUILD_TYPE=Release',
    f'-DBEAM_CUDA_ARCHITECTURES={CUDA_ARCHITECTURES}',
    f"-DBEAM_PUZZLE_INFO_JSON={CUBE4_DATA_DIR / 'puzzle_info.json'}",
    '-DBEAM_ENABLE_LIBTORCH_STREAM1=ON',
    f'-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}',
    f'-DCUTLASS_DIR={CUTLASS_DIR}',
    f'-DBEAM_ENABLE_DEBUG={"ON" if ENABLE_DEBUG else "OFF"}',
    f'-DBEAM_ENABLE_DEPTH_LOGS={"ON" if ENABLE_DEPTH_LOGS else "OFF"}',
    f'-DBEAM_ENABLE_DEBUG_LOGS={"ON" if ENABLE_DEBUG_LOGS else "OFF"}',
    f'-DBEAM_DEBUG_STREAM_TIMING={"ON" if DEBUG_STREAM_TIMING else "OFF"}',
    f'-DBEAM_DEBUG_INFERENCE_TRACE={"ON" if DEBUG_INFERENCE_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_PATH_TRACE={"ON" if DEBUG_PATH_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_FINAL_VALIDATE={"ON" if DEBUG_FINAL_VALIDATE else "OFF"}',
    f'-DBEAM_DEBUG_FINAL_EXCHANGE_TRACE={"ON" if DEBUG_FINAL_EXCHANGE_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_FINAL_HISTOGRAM_TRACE={"ON" if DEBUG_FINAL_HISTOGRAM_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_STREAM4_HISTOGRAM_TRACE={"ON" if DEBUG_STREAM4_HISTOGRAM_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_DEPTH_FLOW_TRACE={"ON" if DEBUG_DEPTH_FLOW_TRACE else "OFF"}',
    f'-DBEAM_DEBUG_PIPELINE_STATS={"ON" if DEBUG_PIPELINE_STATS else "OFF"}',
], cwd=REPO_DIR)
run_checked(['cmake', '--build', BUILD_DIR, '--target', 'production_runner', 'stream1_transformer_cuda_tests', '-j', '2'])
run_checked([BUILD_DIR / 'stream1_transformer_cuda_tests'], cwd=WORK_DIR)


import os
from pathlib import Path
import shutil
import subprocess
import threading
import sys
import time



def monitor_gpus(path, stop_event):
    query = [
        'nvidia-smi',
        '--query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits',
    ]
    with path.open('w', buffering=1, encoding='utf-8') as stream:
        stream.write('sample_wall_sec,timestamp,index,name,utilization_gpu_pct,memory_used_mib,memory_total_mib,power_draw_w\n')
        monitor_start = time.time()
        while not stop_event.is_set():
            result = subprocess.run(query, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            elapsed = time.time() - monitor_start
            for sample in result.stdout.splitlines():
                stream.write(f'{elapsed:.3f},{sample}\n')
            stop_event.wait(5.0)

def safe_name(text):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(text))


def should_print_live(line):
    low = line.lower()
    return (
        'stream1_backend=' in line or
        'stream1_model_backend=' in line or
        'stream1_transformer_dims' in line or
        'runner_phase=' in line or
        'candidate_history_' in line or
        'history_' in line or
        line.startswith('depth ') or
        'puzzle_' in line or
        'error' in low or
        'exception' in low or
        'terminate called' in low or
        'what():' in low or
        'childfailed' in low or
        'traceback' in low or
        'RUN_' in line
    )


def env_for_solve(puzzle_id):
    values = derived_values()
    history_path = Path(HISTORY_DISK_PATH) / f'p{puzzle_id}_beam{BEAM_WIDTH}_d{DEPTH_LIMIT}'
    cleanup_path(history_path)
    history_path.mkdir(parents=True, exist_ok=True)
    nccl_id_file = WORK_DIR / f'beam_solver_nccl_transformer_solve_p{puzzle_id}.bin'
    cleanup_path(nccl_id_file)
    cleanup_path(Path(str(nccl_id_file) + '.tmp'))
    env = os.environ.copy()
    env.update({
        'BEAM_STREAM1_MODE': 'model',
        'BEAM_STREAM1_EXECUTOR': 'libtorch_eager',
         'BEAM_RING_GRAPH_EXECS_PER_LANE': '32',
         'BEAM_STREAM1_TRANSFORMER_MICRO': '384',
         'BEAM_STREAM1_TRANSFORMER_BLOCK51': '1',
         'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY': '1',
         'BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION': '0',
         'BEAM_STREAM1_TRANSFORMER_QKV_POLICY': 'm128n128',
         'BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE': '8',
         'BEAM_STREAM1_TRANSFORMER_FF1_POLICY': 'm128n128w64n32',
         'BEAM_STREAM1_TRANSFORMER_FF1_STAGES': '2',
         'BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE': '1',
         'BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY': 'm128n128',
         'BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE': 'fused',
         'BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE': '2',
         'BEAM_STREAM1_TRANSFORMER_FF2_POLICY': 'm128n128',
         'BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE': 'fused',
         'BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE': '2',
         'BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY': 'row',
         'BEAM_STREAM1_TRANSFORMER_ATTENTION_TILE_POLICY': 'q64k64',
         'BEAM_STREAM1_TRANSFORMER_ATTENTION_MAX_K_POLICY': 'padded64',
        'BEAM_WEIGHT_DIR': str(WEIGHT_OUT_DIR),
        'BEAM_PUZZLE_INFO_JSON': str(CUBE4_DATA_DIR / 'puzzle_info.json'),
        'BEAM_TEST_CSV': str(CUBE4_DATA_DIR / 'test.csv'),
        'BEAM_RUNTIME_CONFIG_MODE': RUNTIME_CONFIG_MODE,
        'BEAM_SHARD_COUNT': str(SHARD_COUNT),
        'BEAM_SHARD_CAPACITY_CANDIDATES': str(values['shard_capacity']),
        'BEAM_SHARD_CAPACITY_SCALE_PPM': str(SHARD_CAPACITY_SCALE_PPM),
        'BEAM_SHARD_BUFFER_COUNT': str(SHARD_BUFFER_COUNT),
        'BEAM_GLOBAL_SPILL_CAPACITY': str(GLOBAL_SPILL_CAPACITY),
        'BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM': str(STREAM5_RECV_CAPACITY_SCALE_PPM),
        'BEAM_GPU_HEADROOM_BYTES': str(GPU_HEADROOM_BYTES),
        'BEAM_B_MICRO': str(B_MICRO),
        'BEAM_STREAM1_CONCURRENCY': str(STREAM1_CONCURRENCY),
        'BEAM_STREAM3_RING_SLOTS': str(STREAM3_RING_SLOTS),
        'BEAM_STREAM4_BATCH_CANDIDATES': str(STREAM4_BATCH_CANDIDATES),
        'BEAM_STREAM4_TRIGGER_CANDIDATES': str(STREAM4_TRIGGER_CANDIDATES),
        'BEAM_STREAM4_ACTIVE_SORT_SLOTS': str(STREAM4_ACTIVE_SORT_SLOTS),
        'BEAM_HISTORY_MODE': str(HISTORY_MODE),
        'BEAM_HISTORY_RAM_BYTES': str(HISTORY_RAM_BYTES),
        'BEAM_HISTORY_DISK_BYTES': str(HISTORY_DISK_BYTES),
        'BEAM_HISTORY_DISK_PATH': str(history_path),
        'BEAM_HISTORY_SLOT_COUNT': str(HISTORY_SLOT_COUNT),
        'BEAM_HISTORY_WORKERS': str(HISTORY_WORKERS),
        'BEAM_SOLVED_NEIGHBORHOOD_RADIUS': str(SOLVED_NEIGHBORHOOD_RADIUS),
        'BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES': str(SOLVED_NEIGHBORHOOD_MAX_ENTRIES),
        'BEAM_STREAM2_SUFFIX_RADIUS': str(STREAM2_SUFFIX_RADIUS),
        'BEAM_STREAM2_SUFFIX_BACKEND': str(STREAM2_SUFFIX_BACKEND),
        'BEAM_STREAM2_SUFFIX_MAX_COUNT': str(STREAM2_SUFFIX_MAX_COUNT),
        'BEAM_DEPTH_LOG_EVERY': str(DEPTH_LOG_EVERY),
        'BEAM_NCCL_ID_FILE': str(nccl_id_file),
    })
    env.pop('WORLD_SIZE', None)
    env.pop('RANK', None)
    env.pop('LOCAL_RANK', None)
    return env, history_path, nccl_id_file


def run_one(puzzle_id):
    env, history_path, nccl_id_file = env_for_solve(puzzle_id)
    log_path = RUN_LOG_DIR / f'torchrun_native_transformer_10m_p{puzzle_id}_d{DEPTH_LIMIT}_b{BEAM_WIDTH}.log'
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        '--no-python',
        f'--nnodes={int(TORCHRUN_NNODES)}',
        f'--nproc-per-node={int(TORCHRUN_NPROC_PER_NODE)}',
        f'--node-rank={int(TORCHRUN_NODE_RANK)}',
        f'--rdzv-backend={TORCHRUN_RDZV_BACKEND}',
        f'--rdzv-endpoint={TORCHRUN_RDZV_ENDPOINT}',
        f'--rdzv-id={TORCHRUN_RDZV_ID}',
        str(BUILD_DIR / 'production_runner_libtorch_stream1'), str(puzzle_id), str(DEPTH_LIMIT), str(BEAM_WIDTH),
    ]
    print('RUN_CUBE4_LIBTORCH_2XT4_START', flush=True)
    print('cmd=', ' '.join(map(str, cmd)), flush=True)
    print('world_size_from_torchrun_shape=', torchrun_world_size(), flush=True)
    print('global_beam=', BEAM_WIDTH, flush=True)
    print('history_path=', history_path, flush=True)
    print('weight_dir=', WEIGHT_OUT_DIR, flush=True)
    print('nccl_id_file=', nccl_id_file, flush=True)
    print('log_path=', log_path, flush=True)
    gpu_monitor_path = RUN_LOG_DIR / f'native_gpu_p{puzzle_id}_b{BEAM_WIDTH}.csv'
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gpus, args=(gpu_monitor_path, stop_monitor), daemon=True)
    monitor_thread.start()
    start = time.time()
    saw_piece_transformer = False
    with log_path.open('w', buffering=1, encoding='utf-8') as log:
        proc = subprocess.Popen(cmd, cwd=REPO_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            if 'stream1_backend=piece_transformer' in line or 'stream1_model_backend=piece_transformer' in line:
                saw_piece_transformer = True
            if should_print_live(line):
                print(line, end='', flush=True)
        rc = proc.wait(timeout=RUN_TIMEOUT_SEC if RUN_TIMEOUT_SEC else None)
    elapsed = time.time() - start
    stop_monitor.set()
    monitor_thread.join(timeout=10)
    print(f'RUN_CUBE4_LIBTORCH_2XT4_RC {rc} seconds={elapsed:.3f}', flush=True)
    if rc != 0:
        raise SystemExit(rc)
    if not saw_piece_transformer:
        raise RuntimeError(f'run log did not contain stream1_backend=piece_transformer or stream1_model_backend=piece_transformer: {log_path}')
    cleanup_path(history_path)
    return {'puzzle_id': puzzle_id, 'return_code': rc, 'seconds': elapsed, 'log_path': str(log_path), 'gpu_monitor': str(gpu_monitor_path)}


results = []
for offset in range(PUZZLE_COUNT):
    results.append(run_one(START_PUZZLE_ID + offset))
print('RUN_CUBE4_LIBTORCH_2XT4_RESULTS=', results, flush=True)
results

