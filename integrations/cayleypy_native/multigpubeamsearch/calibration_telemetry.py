"""Sample GPU thermal and external power-brake events during calibration."""
from __future__ import annotations
import csv
import io
import subprocess
import threading


def parse_telemetry(text):
    rows = []
    for values in csv.reader(io.StringIO(text)):
        if len(values) != 6:
            raise ValueError('unexpected GPU telemetry fields')
        index, uuid, temperature, *events = (value.strip() for value in values)
        if any(event not in ('Active', 'Not Active') for event in events):
            raise ValueError('GPU throttling telemetry unavailable')
        rows.append(dict(index=int(index), uuid=uuid, temperature=int(temperature),
                         throttled=any(event == 'Active' for event in events)))
    if not rows or len({row['index'] for row in rows}) != len(rows):
        raise ValueError('missing or duplicate GPU telemetry')
    return rows


class CalibrationTelemetry:
    """Sampling evidence, not a claim to detect every transient event."""
    def __init__(self, interval=5.0):
        self.interval = interval
        self.samples = []
        self.errors = []
        self.stop = threading.Event()

    def _sample(self):
        query = ('index,uuid,temperature.gpu,clocks_event_reasons.sw_thermal_slowdown,'
                 'clocks_event_reasons.hw_thermal_slowdown,'
                 'clocks_event_reasons.hw_power_brake_slowdown')
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu='+query,
                '--format=csv,noheader,nounits'], capture_output=True, text=True,
                timeout=2, check=True)
            self.samples.append(parse_telemetry(result.stdout))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.append(type(error).__name__)

    def _loop(self):
        while not self.stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=3)
        self._sample()

    @property
    def throttled(self):
        return any(row['throttled'] for sample in self.samples for row in sample)

    def receipt(self):
        return dict(samples=self.samples, errors=self.errors,
                    interval_seconds=self.interval, throttled=self.throttled,
                    scope='sampled visible GPUs; transient events may be missed')
