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
        return store.prepare_sized_for_session(source, cache, run, graph, digest, ['input.fp16'],
                                               [len(manifest), len(tensor)], force)

    legacy = _beam_runtime.MlpSnapshotStore()
    legacy_path = legacy.prepare(source, cache, root / 'legacy', graph, digest, ['input.fp16'], False)[0]
    try:
        legacy.verify_snapshot(legacy_path, graph, digest, ['input.fp16'])
    except RuntimeError as error:
        assert 'no validated file sizes' in str(error)
    else:
        raise AssertionError('unsized legacy snapshot used fast launch verification')

    first = prepare(root / 'run1')
    assert Path(first[0]) == root / 'run1/weights'
    assert first[1:] == (False, len(manifest) + len(tensor))
    store.verify_snapshot(first[0], graph, digest, ['input.fp16'])
    for wrong in (root / 'other',):
        try:
            store.verify_snapshot(wrong, graph, digest, ['input.fp16'])
        except RuntimeError:
            pass
        else:
            raise AssertionError('unowned snapshot was accepted')
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
