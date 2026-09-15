"""Convert the saved Cube4 generator object into the benchmark actions schema."""
import json
from pathlib import Path

info = json.loads(Path('/workspace/puzzle_info.json').read_text())
names = json.loads(Path('/workspace/manifest.json').read_text())['move_names']
actions = [info['generators'][name] for name in names]
assert len(actions) == 24
assert all(sorted(action) == list(range(96)) for action in actions)
Path('/workspace/cube4_actions.json').write_text(json.dumps({'actions': actions}))
