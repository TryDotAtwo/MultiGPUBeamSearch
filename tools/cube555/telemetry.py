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
        with (self.output / 'gpu_samples.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['elapsed_s', 'gpu', 'used_mib', 'total_mib', 'gpu_util_pct', 'power_w'])
            while not self.stop.is_set():
                try:
                    p = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
                    for row in csv.reader(p.stdout.splitlines()):
                        writer.writerow([round(time.monotonic()-start, 3), *[v.strip() for v in row]])
                    f.flush()
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
