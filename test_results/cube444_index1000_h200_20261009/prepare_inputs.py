"""Public Kaggle inputs downloaded inside the remote image build; no credentials."""
import hashlib
import io
import json
import os
import pathlib
import sys
import urllib.request
import zipfile

root = pathlib.Path('/workspace')
sys.path.insert(0, str(root))
os.environ.setdefault('BENCHMARK_DEADLINE', '0')
from sweep import EXPECTED

assets = root / 'inputs/assets'
assets.mkdir(parents=True, exist_ok=True)
url = 'https://www.kaggle.com/api/v1/datasets/download/artgor/cube444-q-beam-tpu-assets'
payload = urllib.request.urlopen(url, timeout=180).read()
with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
    for name, expected in EXPECTED.items():
        entries = [n for n in bundle.namelist() if pathlib.PurePosixPath(n).name == name]
        assert len(entries) == 1, (name, entries)
        content = bundle.read(entries[0])
        assert hashlib.sha256(content).hexdigest() == expected, name
        (assets / name).write_bytes(content)
sys.path.insert(0, str(root / 'source/tools'))
from export_cube444_blend import export
export(assets, root / 'inputs/bundle')
(root / 'results/image_asset_hashes.json').write_text(json.dumps(EXPECTED, indent=2))
print('IMAGE_INPUTS_VERIFIED', flush=True)

