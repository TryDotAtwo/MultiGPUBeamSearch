from unittest.mock import patch
import pytest
from tools.cuda_device_observation import observe

@pytest.mark.parametrize('change',[None,'binary','mapping'])
def test_probe_observation(tmp_path,change):
    probe=tmp_path/'probe'; probe.write_bytes(b'probe')
    calls=[]
    def query(_):
        calls.append(1)
        if len(calls)==2 and change=='binary': probe.write_bytes(b'replaced')
        return {'device':1 if len(calls)==2 and change=='mapping' else 0}
    with patch('tools.cuda_device_observation.query',query):
        if change:
            with pytest.raises(ValueError,match='changed during observation'): observe(probe)
        else:
            assert observe(probe)['production_quality_accepted'] is False
