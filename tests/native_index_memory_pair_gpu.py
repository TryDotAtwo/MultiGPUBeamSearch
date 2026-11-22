"""Repeat full-step pairs while sampling physical device memory through NVML."""
import ctypes
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import native_radix128_pair_gpu as paired


class MemoryInfo(ctypes.Structure):
    _fields_ = [('total', ctypes.c_ulonglong), ('free', ctypes.c_ulonglong),
                ('used', ctypes.c_ulonglong)]


class MemorySampler:
    def __init__(self, world):
        self.library = ctypes.CDLL('libnvidia-ml.so.1')
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)]
        self.library.nvmlDeviceGetMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryInfo)]
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError('NVML initialization failed')
        self.handles = []
        for index in range(world):
            handle = ctypes.c_void_p()
            if self.library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                raise RuntimeError('NVML device lookup failed')
            self.handles.append(handle)
        self.samples = []
        self.errors = []
        self.stop = threading.Event()

    def sample(self):
        try:
            used = []
            for handle in self.handles:
                info = MemoryInfo()
                if self.library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != 0:
                    raise RuntimeError('NVML memory query failed')
                used.append(info.used)
            self.samples.append(dict(monotonic_ns=time.monotonic_ns(), used_bytes=used))
        except Exception as error:
            self.errors.append(type(error).__name__)

    def run(self):
        while not self.stop.wait(0.025):
            self.sample()

    def wait_idle(self):
        deadline = time.monotonic() + 10.0
        while True:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                     '--format=csv,noheader,nounits'], check=True, capture_output=True,
                    text=True, timeout=5)
                processes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            except (OSError, subprocess.SubprocessError) as error:
                self.errors.append('compute process query: ' + type(error).__name__)
                processes = ['query failed']
            self.sample()
            if not processes or self.errors or time.monotonic() >= deadline:
                return processes
            time.sleep(0.1)

    def __enter__(self):
        # The monitoring process may itself change the whole-device baseline.
        # Require no compute processes and compare against that baseline.
        self.initial_processes = self.wait_idle()
        self.start_index = max(0, len(self.samples) - 1)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('memory sampler did not stop')
        self.final_processes = self.wait_idle()
        if self.library.nvmlShutdown() != 0:
            self.errors.append('nvmlShutdown')

    def receipt(self):
        samples = self.samples[self.start_index:]
        if not samples:
            raise RuntimeError('missing NVML memory samples')
        world = len(self.handles)
        first = samples[0]['used_bytes']
        last = samples[-1]['used_bytes']
        peak = [max(sample['used_bytes'][i] for sample in samples) for i in range(world)]
        validation_errors = list(self.errors)
        if len(samples) < 10:
            validation_errors.append('insufficient samples')
        if any(peak[i] - first[i] < 64 * 1024 * 1024 for i in range(world)):
            validation_errors.append('sampler did not observe both rank allocations')
        if self.initial_processes or self.final_processes:
            validation_errors.append('compute process present at a measurement boundary')
        if any(abs(first[i] - last[i]) > 2 * 1024 * 1024 for i in range(world)):
            validation_errors.append('whole-device baseline drifted across measurement')
        return dict(interval_seconds=0.025, initial_used_bytes=first, peak_used_bytes=peak,
                    final_used_bytes=last, initial_compute_processes=self.initial_processes,
                    final_compute_processes=self.final_processes, samples=samples, errors=validation_errors,
                    scope='sampled whole-device NVML memory; unobserved peaks remain possible')


def main():
    world = int(sys.argv[sys.argv.index('--world') + 1]) if '--world' in sys.argv else 2
    out = Path(sys.argv[sys.argv.index('--out') + 1]).resolve()
    original = paired.NativePipelineProbe.measure

    def measure_with_memory(self, *args, **kwargs):
        with MemorySampler(world) as sampler:
            result = original(self, *args, **kwargs)
        receipt = sampler.receipt()
        directory = self.directory / str(self.counter)
        (directory / 'gpu-memory.json').write_text(json.dumps(receipt, indent=2) + '\n')
        if receipt['errors']:
            raise RuntimeError('invalid GPU memory receipt: ' + str(receipt['errors']) +
                               f" initial={receipt['initial_used_bytes']} final={receipt['final_used_bytes']}")
        return result

    paired.NativePipelineProbe.measure = measure_with_memory
    source_paths = (Path(__file__), Path(paired.__file__))
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    paired.main()
    data = json.loads((out / 'paired-results.json').read_text())
    rows = []
    for pair in data['pairs']:
        control = json.loads((Path(pair['control']['run']) / 'gpu-memory.json').read_text())
        candidate = json.loads((Path(pair['radix']['run']) / 'gpu-memory.json').read_text())
        control_net = [peak - baseline for peak, baseline in
                       zip(control['peak_used_bytes'], control['initial_used_bytes'])]
        candidate_net = [peak - baseline for peak, baseline in
                         zip(candidate['peak_used_bytes'], candidate['initial_used_bytes'])]
        rows.append(dict(control_peak_bytes=control['peak_used_bytes'],
                         candidate_peak_bytes=candidate['peak_used_bytes'],
                         control_baseline_bytes=control['initial_used_bytes'],
                         candidate_baseline_bytes=candidate['initial_used_bytes'],
                         control_net_peak_bytes=control_net, candidate_net_peak_bytes=candidate_net,
                         saved_net_bytes=[a - b for a, b in zip(control_net, candidate_net)],
                         speedup_percent=pair['speedup_percent']))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[path.name] for path in source_paths):
        raise RuntimeError('benchmark source changed during measurement')
    result = dict(pairs=rows, median_saved_net_bytes_by_gpu=[statistics.median(row['saved_net_bytes'][i] for row in rows)
                  for i in range(world)], median_speedup_percent=data['median_speedup_percent'],
                  source_sha256=source_hashes,
                  scope='paired sampled whole-device peak above idle baseline; 25 ms sampling can miss peaks')
    (out / 'memory-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('INDEX_MEMORY_PAIRS_OK', flush=True)


if __name__ == '__main__':
    main()
