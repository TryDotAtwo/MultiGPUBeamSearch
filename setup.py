from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import tempfile
import os
from pathlib import Path

# setup.py нужен для стационарной сборки вместо runtime JIT через torch.load().
# Runtime-вариант уже реализован в beam_engine.py.

# Для Kaggle: убедимся, что build directory находится в writable location
build_temp = None
if os.path.exists('/kaggle/working'):
    # Kaggle environment - используем /kaggle/working для build
    build_temp = '/kaggle/working/build'
    Path(build_temp).mkdir(parents=True, exist_ok=True)
else:
    # Local environment - используем default
    build_temp = None

include_dirs = []
for rel_include in ("third_party/cutlass/include", "third_party/cutlass/tools/util/include"):
    if Path(rel_include).exists():
        include_dirs.append(rel_include)

setup(
    name="beam_engine_ext",
    ext_modules=[
        CUDAExtension(
            name="beam_engine_ext",
            sources=["beam_engine.cpp", "beam_kernels.cu", "beam_config.cpp", "beam_memory.cpp", "beam_kernels_stream2.cu", "beam_kernels_final.cu", "beam_kernels_stream3.cu", "beam_kernels_stream4.cu", "beam_dispatcher.cpp"],
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-std=c++17"],
            },
            extra_link_args=["-lnccl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    options={
        "build": {
            "build_base": build_temp,
        } if build_temp else {}
    },
)
