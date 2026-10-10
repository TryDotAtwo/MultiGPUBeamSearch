import os
from pathlib import Path
os.environ.update(GNN_SOURCE='/root/gnn-large-source',GNN_COMMIT='83e46be081cae82c071128713bb93c47178f9cf3',TORCH_CUDA_ARCH_LIST='8.6',CUDA_HOME='/usr/local/cuda',MAX_JOBS='2')
os.environ['PATH']='/venv/main/bin:/usr/local/cuda/bin:'+os.environ.get('PATH','')
code=Path('/root/gnn-large-benchmark.py').read_text()
code=code.replace('probe.write_text(', '(lambda value: None)(')
code=code.replace("artifact=export_gnn(model,contract,out/'artifact')", "from types import SimpleNamespace\nartifact=SimpleNamespace(weights_dir=out/'artifact/weights',artifact_hash=json.loads((out/'result.json').read_text())['metadata']['artifact_hash'])")
code=code.replace('(1,2,4,8,16,24)', '(3,)')
code=code.replace("(out/'result.json').write_text", "(out/'result-extra.json').write_text")
exec(compile(code,'extra-batch-benchmark.py','exec'))
