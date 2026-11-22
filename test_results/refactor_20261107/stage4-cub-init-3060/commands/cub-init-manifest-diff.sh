python - <<'PY'
import json
from pathlib import Path
a=json.loads(Path('/workspace/beam-results/score-bits/default1/source-manifest.json').read_text())
b=json.loads(Path('/workspace/beam-results/cub-init/green1/source-manifest.json').read_text())
for k in sorted(set(a)|set(b)):
 if (k.startswith(('cuda/','src/','native/','rust/')) or k=='CMakeLists.txt') and a.get(k)!=b.get(k):
  print(k,a.get(k),b.get(k))
PY
