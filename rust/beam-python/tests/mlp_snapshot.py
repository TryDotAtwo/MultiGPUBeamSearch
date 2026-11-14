"""Exercise the real PyO3 MLP snapshot owner without CUDA."""
import hashlib
from pathlib import Path
import tempfile

import _beam_runtime


with tempfile.TemporaryDirectory(prefix='beam-pyo3-mlp-') as temp:
    root = Path(temp)
    source, cache = root / 'source', root / 'cache'
    source.mkdir()
    cache.mkdir()
    manifest = b'{"dtype":"fp16"}\n'
    tensor = b'\x00\x3c' * 32769
    (source / 'manifest.json').write_bytes(manifest)
    (source / 'input.fp16').write_bytes(tensor)
    graph = 'a' * 64
    digest = hashlib.sha256(manifest + graph.encode() + b'input.fp16' + tensor).hexdigest()
    store = _beam_runtime.MlpSnapshotStore()

    def prepare(run, force=False):
        return store.prepare(source, cache, run, graph, digest, ['input.fp16'], force)

    first = prepare(root / 'run1')
    assert Path(first[0]) == root / 'run1/weights'
    assert first[1:] == (False, len(manifest) + len(tensor))
    repeated = prepare(root / 'run2')
    assert Path(repeated[0]) == Path(first[0]) and repeated[1:] == (True, 0)
    fresh = prepare(root / 'run2', True)
    assert Path(fresh[0]) == root / 'run2/weights' and not fresh[1]
    (source / 'input.fp16').write_bytes(b'\x01\x3c' + tensor[2:])
    try:
        prepare(root / 'run3')
    except RuntimeError as error:
        assert 'hash mismatch' in str(error)
    else:
        raise AssertionError('modified source was accepted')
    assert not (root / 'run3/weights').exists()
    store.clear()

print('PYO3_MLP_SNAPSHOT_PASS')
