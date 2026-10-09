"""Remote-only build from immutable GitHub evidence; no local product execution."""
import hashlib, io, json, os, pathlib, subprocess, tarfile, time, urllib.request

ROOT = pathlib.Path('/workspace')
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
SOURCE = ROOT / 'source'
SOURCE.mkdir(exist_ok=True)
COMMIT = 'c82649f2d11bdfb2937bc3a0c0435999c302eb2a'
URL = f'https://github.com/TryDotAtwo/MultiGPUBeamSearch/archive/{COMMIT}.tar.gz'
data = urllib.request.urlopen(URL, timeout=90).read()
prefix = f'MultiGPUBeamSearch-{COMMIT}/test_results/cube444_blend_fp16_20261008/'
with tarfile.open(fileobj=io.BytesIO(data)) as archive:
    manifest = json.load(archive.extractfile(prefix + 'source_manifest.json'))
    for name, digest in manifest.items():
        target = SOURCE / name
        assert target.resolve().is_relative_to(SOURCE.resolve())
        content = archive.extractfile(prefix + 'source_snapshot/' + name).read()
        assert hashlib.sha256(content).hexdigest() == digest, name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (OUT / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    for name in ('probe.cpp', 'launch_order.py'):
        (ROOT / name).write_bytes(archive.extractfile(prefix + name).read())
os.environ['PATH'] = '/venv/main/bin:/usr/local/cuda/bin:' + os.environ['PATH']
cutlass = ROOT / 'cutlass'
subprocess.run(['git', 'clone', '--quiet', 'https://github.com/NVIDIA/cutlass.git', str(cutlass)], check=True)
subprocess.run(['git', '-C', str(cutlass), 'checkout', '--quiet', 'afa1772203677c5118fcd82537a9c8fefbcc7008'], check=True)
import torch, nvidia.nccl
nccl = pathlib.Path(next(iter(nvidia.nccl.__path__)))
args = ['cmake', '-S', str(SOURCE), '-B', str(ROOT / 'build'), '-DCMAKE_BUILD_TYPE=Release',
        '-DBEAM_CUDA_ARCHITECTURES=86', '-DBEAM_STATE_LOGICAL_BYTES=96', '-DBEAM_STATE_ALIGNMENT=16',
        '-DBEAM_MOVE_COUNT=24', '-DCUTLASS_DIR=' + str(cutlass), '-DBEAM_ENABLE_LIBTORCH_STREAM1=ON',
        '-DBEAM_ENABLE_DEBUG=ON', '-DBEAM_ENABLE_DEBUG_LOGS=ON', '-DBEAM_ENABLE_DEPTH_LOGS=ON',
        '-DCMAKE_PREFIX_PATH=' + torch.utils.cmake_prefix_path, '-DNCCL_INCLUDE_DIR=' + str(nccl / 'include'),
        '-DNCCL_LIBRARY=' + str(nccl / 'lib/libnccl.so.2')]
subprocess.run(args, check=True)
subprocess.run(['cmake', '--build', str(ROOT / 'build'), '--target', 'production_runner_libtorch_stream1', '-j', '8'], check=True)
binary = ROOT / 'build/production_runner_libtorch_stream1'
frozen = ROOT / 'runner-frozen'
frozen.write_bytes(binary.read_bytes())
frozen.chmod(0o555)
(OUT / 'build_receipt.json').write_text(json.dumps({'status': 'PASS', 'github_commit': COMMIT,
    'source_files_verified': len(manifest), 'binary_sha256': hashlib.sha256(frozen.read_bytes()).hexdigest(),
    'torch': torch.__version__, 'cuda': torch.version.cuda}, indent=2))
print('BUILD_PASS', flush=True)

