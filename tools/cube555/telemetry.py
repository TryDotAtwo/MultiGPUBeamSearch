"""Bounded, process-independent NVML telemetry; never attributes a sample as a peak allocation."""
import csv
import json
import re
import subprocess
import threading
import time
from pathlib import Path


def host_memory_sample(root=Path('/sys/fs/cgroup'), proc_root=Path('/proc/self')):
    """Memory controller samples; scope is unknown until membership is resolved."""
    sample = {}
    v2_root, v1_root = root, root / 'memory'
    scope = 'controller_root_unverified'
    try:
        memberships = {}
        for line in (proc_root / 'cgroup').read_text().splitlines():
            _, controllers, member = line.split(':', 2)
            if not controllers:
                memberships['cgroup2'] = member
            elif 'memory' in controllers.split(','):
                memberships['cgroup'] = member
        for line in (proc_root / 'mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            fields, filesystem = before.split(), after.split()
            kind = filesystem[0]
            if kind not in memberships or (kind == 'cgroup' and
                    'memory' not in filesystem[-1].split(',')):
                continue
            mount_root = Path(fields[3])
            member = Path(memberships[kind])
            try:
                relative = member.relative_to(mount_root)
            except ValueError:
                continue
            candidate = Path(fields[4]) / relative
            marker = 'memory.current' if kind == 'cgroup2' else 'memory.usage_in_bytes'
            if (candidate / marker).is_file():
                if kind == 'cgroup2':
                    v2_root = candidate
                else:
                    v1_root = candidate
                scope = 'process_cgroup'
                break
    except (OSError, ValueError, IndexError):
        pass
    for name in ('memory.current', 'memory.max', 'memory.peak'):
        try:
            value = (v2_root / name).read_text().strip()
            sample[name] = int(value) if value != 'max' else 'unlimited'
        except (OSError, ValueError):
            pass
    try:
        sample['memory.events'] = {
            key: int(value) for key, value in
            (line.split() for line in (v2_root / 'memory.events').read_text().splitlines())
        }
    except (OSError, ValueError):
        pass
    if not sample:
        # Molab currently exposes cgroup v1; /proc/meminfo may describe the
        # larger host, so it cannot substitute for this container limit.
        memory_root = v1_root
        for legacy, field in (
            ('memory.usage_in_bytes', 'memory.current'),
            ('memory.limit_in_bytes', 'memory.max'),
            ('memory.max_usage_in_bytes', 'memory.peak'),
            ('memory.failcnt', 'memory.failcnt'),
        ):
            try:
                sample[field] = int((memory_root / legacy).read_text().strip())
            except (OSError, ValueError):
                pass
        try:
            sample['memory.oom_control'] = {
                key: int(value) for key, value in
                (line.split() for line in (memory_root / 'memory.oom_control').read_text().splitlines())
            }
        except (OSError, ValueError):
            pass
    if sample:
        sample['memory.scope'] = scope
        sample['memory.note'] = 'Own controller limit may also be constrained by ancestor cgroups.'
    return sample


class Telemetry:
    def __init__(self, output):
        self.output = output
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        start = time.monotonic()
        last_report = -30.0
        with (self.output / 'gpu_samples.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_s', 'gpu', 'used_mib', 'total_mib', 'gpu_util_pct', 'power_w',
                             'sm_clock_mhz', 'memory_clock_mhz', 'temperature_c', 'pstate'])
            while not self.stop.is_set():
                try:
                    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw,clocks.sm,clocks.mem,temperature.gpu,pstate', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                    for row in csv.reader(p.stdout.splitlines()):
                        writer.writerow([round(time.monotonic()-start, 3), *[v.strip() for v in row]])
                    f.flush()
                    elapsed = time.monotonic() - start
                    host_memory = host_memory_sample()
                    with (self.output / 'host_memory_samples.jsonl').open('a') as host_log:
                        host_log.write(json.dumps(dict(elapsed_s=round(elapsed, 3),
                                                       **host_memory)) + '\n')
                    if elapsed - last_report >= 30:
                        last_report = elapsed
                        stage = 'initializing'
                        status_path = self.output / 'run_summary.json'
                        if status_path.exists():
                            try:
                                status = json.loads(status_path.read_text())
                                stage = status.get('status', stage)
                            except (OSError, ValueError):
                                pass
                        print(f'[progress] elapsed={elapsed:.0f}s stage={stage} '
                              f'GPU(index,MiB,totalMiB,util%,W,SMMHz,memMHz,C,pstate)={p.stdout.strip().replace(chr(10), " | ")}', flush=True)
                        print('[host-memory] '+json.dumps(host_memory), flush=True)
                        # Native depth logs can be quiet during a long full beam layer.
                        # Print only known progress records, never arbitrary log contents.
                        for log in sorted(self.output.rglob('stdout.log')):
                            try:
                                with log.open('rb') as stream:
                                    stream.seek(max(0, log.stat().st_size - 8192))
                                    tail = stream.read().decode('utf-8', errors='replace')
                                lines = [line for line in tail.splitlines() if
                                         re.search(r'depth_done=|puzzle_solved=|collection_truncated=|solved_neighborhood_entries=', line)]
                                if lines:
                                    print(f'[rank-progress {log.parent.name}] {lines[-1]}', flush=True)
                            except OSError:
                                pass
                except (OSError, subprocess.SubprocessError) as e:
                    (self.output / 'telemetry_error.txt').write_text(str(e))
                    return
                self.stop.wait(1)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=7)
        peaks = {}
        path = self.output / 'gpu_samples.csv'
        if path.exists():
            for row in csv.DictReader(path.open()):
                gpu = row['gpu']
                peaks[gpu] = max(peaks.get(gpu, 0), float(row['used_mib']))
        depths = []
        for p in self.output.rglob('rank-0.log'):
            for d, sec in re.findall(r'depth_done=(\d+) depth_sec=([\d.e+-]+)', p.read_text(errors='replace')):
                depths.append(dict(log=str(p.relative_to(self.output)), depth=int(d), seconds=float(sec)))
        truncations = []
        for p in self.output.rglob('rank-*.log'):
            for depth, rank, hits, stored, dropped in re.findall(
                    r'collection_truncated=1 depth=(\d+) rank=(\d+) hits=(\d+) stored=(\d+) dropped=(\d+)',
                    p.read_text(errors='replace')):
                truncations.append(dict(log=str(p.relative_to(self.output)), depth=int(depth),
                    rank=int(rank), hits=int(hits), stored=int(stored), dropped=int(dropped)))
        report = dict(collection_truncated=bool(truncations), collection_truncations=truncations,
                      collection_dropped_hits=sum(row['dropped'] for row in truncations),
                      sample_interval_seconds=1, sampled_peak_gpu_mib=peaks,
                      note='Sampled device memory includes native CUDA and LibTorch; brief peaks may be missed.', depths=depths)
        (self.output / 'performance.json').write_text(json.dumps(report, indent=2))
        return report
