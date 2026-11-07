"""Remote-only public API acceptance: strict dispatch and replay evidence."""
import json
import os
from pathlib import Path
from cayleypy import CayleyGraph, PermutationGroups
from cayleypy_native import NativeOptions, beam_search

root = Path('/workspace/api-source')
results = []
for executor in ('cutlass','libtorch'):
    for world in (1,2):
        graph = CayleyGraph(PermutationGroups.lrx(4), device='cuda', num_gpus=1)
        options = NativeOptions(source_dir=root, cache_dir=Path('/workspace/api-cache'),
                                num_gpus=world, inference_backend=executor)
        result = beam_search(graph, native_options=options, start_state=[3,2,1,0],
                             beam_width=64,max_steps=16,return_path=True)
        assert result.backend == 'native' and result.path_found
        assert result.native_metadata['devices'] == list(range(world))
        assert result.native_metadata['replay_valid'] is True
        assert result.path_length > 1
        assert result.native_metadata['inference_backend'] == executor
        assert result.native_metadata.get('execution') not in ('already_at_goal','zero_depth_budget')
        results.append({'executor':executor,'world':world,'path':result.path,
                        'metadata':result.native_metadata})
        Path('/workspace/api-acceptance.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
