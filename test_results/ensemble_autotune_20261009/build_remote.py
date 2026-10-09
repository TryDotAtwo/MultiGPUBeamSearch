"""Remote-only build of ensemble admission probes."""
import pathlib, urllib.request, subprocess, torch, json
from torch.utils.cpp_extension import include_paths, library_paths
root=pathlib.Path('/workspace/source')
base='https://raw.githubusercontent.com/TryDotAtwo/MultiGPUBeamSearch/2b36d1824e195acd0ab2fe7922a3476258486583/'
names=['cuda/stream1_blend_readout.hpp','cuda/stream1_ensemble_readout.cu','tools/stream1_ensemble_libtorch.hpp','tools/stream1_ensemble_benchmark.cpp','tools/cube444_blend_libtorch.hpp','tools/stream1_mlp_libtorch_backend.hpp','tools/stream1_transformer_libtorch_backend.hpp']
for name in names: (root/name).write_bytes(urllib.request.urlopen(base+name).read())
defines=['-DBEAM_HAS_CUTLASS=1','-DBEAM_STATE_LOGICAL_BYTES=96','-DBEAM_STATE_PHYSICAL_BYTES=112','-DBEAM_STATE_ALIGNMENT=16','-DBEAM_MOVE_COUNT=24']
with open('/workspace/results/ensemble-model-build.log','w') as log:
 subprocess.run(['nvcc','-std=c++17','-O3','-arch=sm_86',*defines,'-I/workspace/source/src','-I/workspace/cutlass/include','-c',str(root/'cuda/stream1_ensemble_readout.cu'),'-o','/workspace/ensemble-readout.o'],stdout=log,stderr=log,check=True)
 command=['c++','-std=c++17','-O2',*defines,'-D_GLIBCXX_USE_CXX11_ABI='+str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),*[f'-I{p}' for p in include_paths(device_type='cuda')],'-I/workspace/source/src',str(root/'tools/stream1_ensemble_benchmark.cpp'),'/workspace/ensemble-readout.o',*[f'-L{p}' for p in library_paths(device_type='cuda')],*[f'-Wl,-rpath,{p}' for p in library_paths(device_type='cuda')],'-ltorch','-ltorch_cpu','-ltorch_cuda','-lc10','-lc10_cuda','-lcudart','-o','/workspace/ensemble-benchmark']
 subprocess.run(command,stdout=log,stderr=log,check=True)
bundle=pathlib.Path('/workspace/inputs/bundle')
out=pathlib.Path('/workspace/ensemble');out.mkdir(exist_ok=True)
manifest={'schema_version':1,'state_len':96,'output_dim':24,'score_semantics':'distance_q','models':[{'family':'cube444_transformer','weights_dir':str(bundle),'coefficient':0.6},{'family':'cube444_mlp','weights_dir':str(bundle),'coefficient':0.2},{'family':'cube444_mlp','weights_dir':str(bundle),'coefficient':0.2}]}
(out/'ensemble.json').write_text(json.dumps(manifest))
print(json.dumps({'status':'COMPILED','models':3,'torch':torch.__version__}),flush=True)
