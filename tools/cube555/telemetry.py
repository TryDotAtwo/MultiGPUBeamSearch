"""Bounded, process-independent NVML telemetry; never attributes a sample as a peak allocation."""
import csv
import json
import re
import subprocess
import threading
import time


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
            writer.writerow(['elapsed_s', 'gpu', 'used_mib', 'total_mib', 'gpu_util_pct', 'power_w'])
            while not self.stop.is_set():
                try:
                    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                    for row in csv.reader(p.stdout.splitlines()):
                        writer.writerow([round(time.monotonic()-start, 3), *[v.strip() for v in row]])
                    f.flush()
                    elapsed = time.monotonic() - start
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
                              f'GPU(index,MiB,totalMiB,util%,W)={p.stdout.strip().replace(chr(10), " | ")}', flush=True)
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
