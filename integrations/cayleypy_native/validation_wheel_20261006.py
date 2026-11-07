"""Remote-only installed-wheel public API validation and error gates."""
import json
from pathlib import Path
import cayleypy_native
from cayleypy import CayleyGraph, PermutationGroups, Predictor
from cayleypy_native import enable_native, disable_native, NativeOptions, NativeUnavailable

assert '/public-api/' in cayleypy_native.__file__, cayleypy_native.__file__
graph = CayleyGraph(PermutationGroups.lrx(4), device='cuda', num_gpus=1)
start = [3,2,1,0]
records = []
for world in (1,2):
    enable_native(NativeOptions(num_gpus=world, cache_dir=Path('/workspace/api-cache')))
    for explicit in (False, True):
        kwargs = {'predictor':Predictor(graph,'hamming')} if explicit else {}
        result = graph.beam_search(start_state=start,beam_width=64,max_steps=16,return_path=True,**kwargs)
        assert result.backend == 'native' and result.path_found and result.path_length > 1
        assert result.native_metadata['inference_backend'] == 'cutlass'
        assert result.native_metadata['replay_valid'] is True
        records.append({'world':world,'explicit_hamming':explicit,'path':result.path,
                        'path_string':result.get_path_as_string(), 'metadata':result.native_metadata})
    ordinary = graph.beam_search(start_state=start,beam_width=64,max_steps=16,return_path=True,backend='torch')
    assert ordinary.path_found and ordinary.path is not None
    assert graph.beam_search(start_state=start,max_steps=0).path_found is False
    assert graph.beam_search(start_state=[0,1,2,3]).path == []
    disable_native()
enable_native(NativeOptions(num_gpus=3,cache_dir=Path('/workspace/api-cache')))
try:
    graph.beam_search(start_state=start,beam_width=64,max_steps=16)
except NativeUnavailable as error:
    assert 'exceeds' in str(error)
else:
    raise AssertionError('requested GPU count was silently reduced')
disable_native()
Path('/workspace/api-wheel-acceptance.json').write_text(json.dumps({
    'module':cayleypy_native.__file__,'records':records,'gpu_count_rejection':True,
    'ordinary_cayleypy':True,'zero_budget':True,'already_at_goal':True},indent=2))
print('INSTALLED_WHEEL_PUBLIC_API_PASS',flush=True)
