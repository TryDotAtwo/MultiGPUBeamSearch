"""Compare terminal parsers on valid and malformed, bounded receipts."""
import json
from pathlib import Path
import sys

from multigpubeamsearch import _beam_runtime, backend
from multigpubeamsearch.contracts import GraphContract

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
graph = dict(center=[0, 1, 2, 3], generators=[[1, 0, 2, 3], [0, 1, 3, 2]])
start = [1, 0, 2, 3]
contract = GraphContract(4, 4, tuple(map(tuple, graph['generators'])), ('a', 'b'),
                         tuple(graph['center']), tuple(start), 'a' * 64)
head = 'GLOBAL_BEAM_WIDTH_EFFECTIVE=64\n'
solved = 'puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m0\n'
unsolved = 'puzzle_solved=0 puzzle_id=0 seconds=0.2 solution_length=-1 solution=\n'
collect = ('solve_bucket_solution=1 puzzle_id=0 found_depth=1 owner_rank=0 solution_length=1 solution=m0\n'
           'solve_bucket_solution=1 puzzle_id=0 found_depth=2 owner_rank=1 solution_length=3 solution=m1.m1.m0\n'
           'solve_bucket_summary=1 puzzle_id=0 records=2 result_tsv=out.tsv\n'
           'collection_status=depth_reached\n')
valid = {
    'solved': head + solved,
    'unsolved-two-ranks': head + unsolved + unsolved,
    'batch-ranks': head + 'B_MICRO=32\nB_MICRO=64\nSTREAM1_ROWS_PER_JOB=96\nSTREAM3_BATCH_CANDIDATES=2048\n' + solved,
    'collected': head + collect + solved,
    'empty-collection': head + 'solve_bucket_summary=1 puzzle_id=0 records=0 result_tsv=out.tsv\n' + unsolved,
}
invalid = {
    'missing-terminal': head,
    'inconsistent-width': head + 'GLOBAL_BEAM_WIDTH_EFFECTIVE=65\n' + solved,
    'zero-width': 'GLOBAL_BEAM_WIDTH_EFFECTIVE=0\n' + solved,
    'contradictory': head + solved + unsolved,
    'bad-length': head + solved.replace('solution_length=1', 'solution_length=2'),
    'bad-generator': head + solved.replace('solution=m0', 'solution=m2'),
    'bad-replay': head + solved.replace('solution=m0', 'solution=m1'),
    'nan-seconds': head + solved.replace('seconds=0.1', 'seconds=nan'),
    'duplicate-collection': head + collect.replace('solution=m1.m1.m0', 'solution=m0').replace('solution_length=3', 'solution_length=1').replace('found_depth=2', 'found_depth=1') + solved,
}
stricter = {
    'native-error': head + 'production_runner_error=1\n' + solved,
    'invalid-found-depth': head + collect.replace('found_depth=2', 'found_depth=4') + solved,
}


def parse(name, content):
    path = root / (name + '.log'); path.write_text(content, encoding='utf-8')
    outputs = []
    for parser in ('python', 'rust'):
        try:
            if parser == 'python':
                p, width, metadata = backend.parse_terminal(path, contract, max_steps=4)
                value = dict(path=None if p is None else list(p), effective_beam_width=width,
                             metadata=metadata)
            else:
                value = json.loads(_beam_runtime.parse_terminal(path, json.dumps(graph), start, 4))
            outputs.append(dict(ok=True, value=value))
        except Exception as error:
            outputs.append(dict(ok=False, error=type(error).__name__))
    return outputs


records = {}
for name, content in valid.items():
    a, b = parse(name, content); assert a['ok'] and b['ok'] and a['value'] == b['value'], (name, a, b)
    records[name] = [a, b]
for name, content in invalid.items():
    a, b = parse(name, content); assert not a['ok'] and not b['ok'], (name, a, b)
    records[name] = [a, b]
for name, content in stricter.items():
    a, b = parse(name, content); assert a['ok'] and not b['ok'], (name, a, b)
    records[name] = [a, b]
(root / 'results.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
print(f'TERMINAL_DIFFERENTIAL_OK valid={len(valid)} invalid={len(invalid)} stricter={len(stricter)}')
