import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

PROJECT_DIR = pathlib.Path('/kaggle/working/CayleyBeam100H100')


def now():
    return datetime.now().strftime('%H:%M:%S')


def banner(title):
    print('\n' + '=' * 96, flush=True)
    print(f'[{now()}] {title}', flush=True)
    print('=' * 96, flush=True)


def gpu_snapshot():
    try:
        out = subprocess.check_output([
            'nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'
        ], text=True, stderr=subprocess.STDOUT).strip()
        return out.replace('\n', ' | ')
    except Exception as exc:
        return f'nvidia-smi-unavailable:{type(exc).__name__}:{exc}'


def reader(pipe, q):
    try:
        for line in iter(pipe.readline, ''):
            q.put(line.rstrip('\n'))
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def run_live(cmd, *, env, cwd, title, heartbeat_sec=30, timeout_sec=None):
    banner(title)
    print(f'[{now()}] cmd={cmd}', flush=True)
    merged = os.environ.copy()
    merged.update({str(k): str(v) for k, v in env.items()})
    merged.setdefault('PYTHONUNBUFFERED', '1')
    start = time.time()
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=merged, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    q = queue.Queue()
    threading.Thread(target=reader, args=(proc.stdout, q), daemon=True).start()
    lines = []
    last_heartbeat = 0.0
    last_output = time.time()
    while True:
        drained = False
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            drained = True
            last_output = time.time()
            lines.append(line)
            print(line, flush=True)
        rc = proc.poll()
        elapsed = time.time() - start
        if timeout_sec is not None and elapsed > timeout_sec and rc is None:
            proc.kill()
            raise TimeoutError(f'timeout after {elapsed:.1f}s: {cmd}')
        if rc is not None:
            while True:
                try:
                    line = q.get_nowait()
                except queue.Empty:
                    break
                lines.append(line)
                print(line, flush=True)
            print(f'[{now()}] process_exit | returncode={rc} | elapsed_sec={elapsed:.1f}', flush=True)
            text = '\n'.join(lines)
            if rc != 0:
                raise subprocess.CalledProcessError(rc, cmd, output=text)
            return text
        if time.time() - last_heartbeat >= heartbeat_sec:
            print(f'[{now()}] still_running | elapsed_sec={elapsed:.1f} | silent_for_sec={time.time()-last_output:.1f} | gpu=[{gpu_snapshot()}]', flush=True)
            last_heartbeat = time.time()
        time.sleep(0.2 if drained else 0.5)


def prepare_project():
    if PROJECT_DIR.exists():
        shutil.rmtree(PROJECT_DIR)
    input_root = pathlib.Path('/kaggle/input')
    candidates = [p for p in input_root.rglob('*') if p.is_dir() and (p / 'beam_engine.py').exists()]
    assert candidates, 'project dataset with beam_engine.py not found'
    src = candidates[0]
    shutil.copytree(src, PROJECT_DIR)
    for zip_path in list(PROJECT_DIR.glob('*.zip')):
        shutil.unpack_archive(str(zip_path), PROJECT_DIR / zip_path.stem)
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    print('DATASET_DIR_SELECTED', src, flush=True)
    print('PROJECT_FILES', sorted(x.name for x in PROJECT_DIR.iterdir())[:30], flush=True)


def base_env():
    return {
        'PYTHONUNBUFFERED': '1',
        'CUDA_VISIBLE_DEVICES': '0,1',
        'USE_CUDA_GRAPHS': '1',
        'INFERENCE_BACKEND': 'torchscript_ensemble',
        'INFERENCE_PARALLELISM': '2',
        'HISTORY_BACKEND': 'cpu',
        'CPU_HISTORY_WORKERS': '2',
        'B_MICRO': '8192',
        'K_EXPAND_TILE': '16384',
        'SCORE_RING_DEPTH': '8',
        'NET_RING_DEPTH': '2',
        'BUCKET_CAP_PER_PEER': '65536',
        'BETA': '1.20',
        'HASH_LOAD_FACTOR': '0.35',
        'PROBE_LIMIT': '512',
        'HISTOGRAM_PERIOD_MICRO': '2',
        'SUBMISSION_APPEND_EACH': '1',
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_SOCKET_IFNAME': 'lo',
        'GLOO_SOCKET_IFNAME': 'lo',
        'NCCL_DEBUG': 'WARN',
        'BUILD_VERBOSE': '0',
        'LOG_EVERY': '25',
        'DEPTH_LOG_EVERY': '0',
    }


def torchrun(env, title, timeout_sec=None):
    return run_live([
        sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=2', 'scripts/solve_testcsv_2gpu.py'
    ], env=env, cwd=PROJECT_DIR, title=title, heartbeat_sec=30, timeout_sec=timeout_sec)


def main():
    prepare_project()
    env = base_env()
    run_live([sys.executable, '-u', 'scripts/t4_sizing.py'], env={**env, 'GLOBAL_BEAM_WIDTH': str(2**18), 'MAX_DEPTH': '100'}, cwd=PROJECT_DIR, title='CPU history sizing check', timeout_sec=120)

    resume_dir = '/kaggle/working/cpu_history_resume_smoke'
    shutil.rmtree(resume_dir, ignore_errors=True)
    env1 = {**env, 'CPU_HISTORY_CHECKPOINT': '1', 'CPU_HISTORY_DIR': resume_dir, 'RESUME_BEAMSEARCH': '0', 'KNOWN_SCRAMBLE': 'U,R', 'GLOBAL_BEAM_WIDTH': str(2**14), 'MAX_DEPTH': '1', 'SUBMISSION_PATH': '/kaggle/working/resume_part1.csv', 'DEPTH_LOG_EVERY': '1', 'LOG_EVERY': '1'}
    out1 = torchrun(env1, 'resume smoke part1 depth1 checkpoint', timeout_sec=1200)
    assert 'SUBMISSION_WRITTEN' in out1
    env2 = {**env1, 'RESUME_BEAMSEARCH': '1', 'MAX_DEPTH': '2', 'SUBMISSION_PATH': '/kaggle/working/resume_part2.csv'}
    out2 = torchrun(env2, 'resume smoke part2 continue to depth2', timeout_sec=1200)
    assert 'found": true' in out2 or '"found": true' in out2
    print('CPU_HISTORY_RESUME_SMOKE_OK', flush=True)

    matrix = [
        ('beam_2_18_count20', 2**18, 20, '1.20'),
        ('beam_2_16_count50', 2**16, 50, '32.0'),
        ('beam_2_14_count100', 2**14, 100, '32.0'),
        ('beam_2_12_count1001', 2**12, 1001, '32.0'),
    ]
    results = []
    for label, beam, count, beta in matrix:
        out_path = f'/kaggle/working/submission_{label}.csv'
        env_run = {**env, 'CPU_HISTORY_CHECKPOINT': '0', 'RESUME_BEAMSEARCH': '0', 'GLOBAL_BEAM_WIDTH': str(beam), 'MAX_DEPTH': '100', 'TEST_START': '0', 'TEST_COUNT': str(count), 'SUBMISSION_PATH': out_path, 'LOG_EVERY': '25', 'BETA': beta}
        out = torchrun(env_run, f'matrix {label}', timeout_sec=None)
        assert 'SUBMISSION_WRITTEN' in out
        rows = pathlib.Path(out_path).read_text(encoding='utf-8').splitlines()
        assert len(rows) == count + 1, (label, len(rows), count)
        results.append({'label': label, 'rows': len(rows) - 1, 'path': out_path})
    print('CPU_HISTORY_SHARED_SCORER_MATRIX_OK ' + json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
