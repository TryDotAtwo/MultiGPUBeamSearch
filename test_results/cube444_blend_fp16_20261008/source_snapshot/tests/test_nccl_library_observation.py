from hashlib import sha256
from unittest.mock import patch

import pytest

from tools.nccl_library_observation import observe


@pytest.mark.parametrize('change', [None, 'library', 'runner', 'version'])
def test_observation_detects_persistent_change(tmp_path, change):
    runner = tmp_path / 'runner'
    library = tmp_path / 'libnccl.so.2'
    runner.write_bytes(b'runner')
    library.write_bytes(b'library')
    info = dict(schema_version=1, nccl_version=22809,
                nccl_library_path=str(library), communicator_initialized=False)
    calls = []

    def query(_):
        calls.append(1)
        if len(calls) == 2:
            if change in ('library', 'runner'):
                (library if change == 'library' else runner).write_bytes(b'changed')
            elif change == 'version':
                return {**info, 'nccl_version': 22501}
        return info

    with patch('tools.nccl_library_observation.query', query):
        if change:
            with pytest.raises(ValueError, match='changed during observation'):
                observe(runner)
        else:
            result = observe(runner)
            assert result['nccl_library_sha256'] == sha256(b'library').hexdigest()
            assert result['production_quality_accepted'] is False
