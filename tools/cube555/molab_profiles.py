"""Experimental SM120 anchors; measurements, not timing extrapolation, promote rows."""
import json
from pathlib import Path

MAX_BEAM = 17_000_000
ANCHORS = (1_048_576, 2_097_152, 4_000_000, 8_000_000,
           12_000_000, 16_000_000, MAX_BEAM)


def candidate(beam):
    if type(beam) is not int or not 1 <= beam <= MAX_BEAM:
        raise ValueError(f'Molab beam must be in [1, {MAX_BEAM}]')
    registry = json.loads((Path(__file__).resolve().parents[2] /
                           'configs/cube555_molab_profiles.json').read_text())
    return next(row for row in registry['profiles'] if beam <= row['anchor_beam'])
