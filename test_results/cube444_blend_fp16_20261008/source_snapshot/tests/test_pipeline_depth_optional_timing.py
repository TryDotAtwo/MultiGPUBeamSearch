"""Run actual benchmark parser block against literal release/debug depth logs."""
from pathlib import Path
import textwrap
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('timers', [False, True])
def test_real_depth_parser_keeps_absent_timers_unavailable(timers, tmp_path):
    # Regression: release binary output raises KeyError in the debug-only parser.
    text = (ROOT / 'tools/beam_pipeline_ab_probe.py').read_text()
    start = text.index('            rows=[]')
    end = text.index('            if [row[', start)
    actual = textwrap.dedent(text[start:end])
    log = tmp_path / 'stdout.log'
    line = 'depth_done=7 depth_sec=2.5 next_frontier_size=65536 stream4_jobs=9'
    if timers:
        line += ' stream4_ms=12.5 stream12_ms=100.0 stream4_busy_max=2 stream4_pending_max=3'
    log.write_text(line + '\n')
    scope = {'logs': [log]}
    exec(compile(actual, 'actual_depth_parser', 'exec'), scope)
    row = scope['rows'][0]
    assert (row['depth'], row['seconds'], row['frontier'], row['stream4_jobs']) == (7, 2.5, 65536, 9)
    assert row['stream4_ms'] == (12.5 if timers else None)
    assert row['stream12_ms'] == (100.0 if timers else None)
    assert row['stream4_busy_max'] == (2 if timers else None)
    assert row['stream4_pending_max'] == (3 if timers else None)
