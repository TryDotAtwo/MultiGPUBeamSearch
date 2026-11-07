"""Remote-only compile and exhaustive independent LibTorch numeric oracle."""
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
import torch
from cayleypy_native.hamming import write_hamming_artifact

root = Path('/workspace/api-source')
project = Path('/workspace/api-oracle-project')
project.mkdir(exist_ok=True)
(project/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.22)
project(api_libtorch_oracle LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)
add_executable(oracle /workspace/api-source/integrations/cayleypy_native/validation_libtorch_oracle.cpp)
target_include_directories(oracle PRIVATE /workspace/api-source/tools ${TORCH_INCLUDE_DIRS})
target_link_libraries(oracle PRIVATE ${TORCH_LIBRARIES})
set_target_properties(oracle PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
''')
build = Path('/workspace/api-oracle-build')
env = dict(os.environ,TORCH_CUDA_ARCH_LIST='8.6',CUDACXX='/usr/local/cuda/bin/nvcc')
subprocess.run(['cmake','-S',str(project),'-B',str(build),
                '-DCMAKE_PREFIX_PATH='+torch.utils.cmake_prefix_path,
                '-DCMAKE_CUDA_ARCHITECTURES=86'],env=env,check=True)
subprocess.run(['cmake','--build',str(build),'--parallel','2'],env=env,check=True)
for i, center in enumerate(((0,1,2,3),(3,1,0,2),(2,0,2,1))):
    contract = SimpleNamespace(state_len=len(center),num_classes=max(center)+1,center=center,graph_hash='a'*64)
    directory = Path('/workspace') / f'api-oracle-case-{i}'
    weights = write_hamming_artifact(contract,directory)
    subprocess.run([str(build/'oracle'),str(weights),','.join(map(str,center))],check=True)
print('LIBTORCH_NUMERIC_ORACLE_PASS',flush=True)
