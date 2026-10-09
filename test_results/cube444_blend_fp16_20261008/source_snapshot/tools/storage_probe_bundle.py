"""Owned bounded probe preparation/observation; never production admission."""
import hashlib
import json
import math
from pathlib import Path
import stat
import subprocess
from tools.cube4_numeric_gate import unique
from tools.graph_parameter_layout import validate_parameter_storage
from tools.attention_storage_contract import validate_attention_storage_table
from tools.attention_parameter_storage import validate_attention_parameter_storage
from tools.gemm_parameter_storage import validate_gemm_parameter_storage
from tools.graph_gemm_iterator_contract import validate_gemm_iterator_geometry
from tools.public_parameter_storage import (validate_gemm_public_storage,
    validate_attention_public_storage,validate_nested_gemm_storage,observe_public_layout_headers)

PROBES={
    'parameter':('stream1_parameter_storage_probe',validate_parameter_storage),
    'attention_geometry':('stream1_attention_storage_probe',validate_attention_storage_table),
    'attention_parameter':('stream1_attention_parameter_probe',validate_attention_parameter_storage),
    'gemm_parameter':('stream1_gemm_parameter_probe',validate_gemm_parameter_storage),
    'gemm_iterator':('stream1_gemm_iterator_probe',validate_gemm_iterator_geometry),
    'gemm_public_fields':('stream1_gemm_field_ledger_probe',validate_gemm_public_storage),
    'attention_public_fields':('stream1_attention_field_ledger_probe',validate_attention_public_storage),
    'gemm_nested_storage':('stream1_gemm_nested_storage_probe',validate_nested_gemm_storage)}

MAINLOOP_LAYOUT_SHA256={
    'predicated_tile_access_iterator.h':'e24899b2f01c7ac9c6cec617c4bb20b17d8dae139bf06bb8ef0f624ffb38b64f',
    'predicated_tile_iterator.h':'894c373193b06e6592aec5403fc585bc6c6bc67a492a81a035f4cb89059e9557',
    'predicated_tile_access_iterator_params.h':'10ee5d4924f625e2c465311a9a821ab1821d020a7ab09ae27b784894ae09a768'}


def observe_mainloop_layout(build):
    cache=(Path(build)/'CMakeCache.txt').read_bytes()
    if len(cache)>1024*1024:raise ValueError('mainloop CMake cache exceeds bound')
    paths=[line.split('=',1)[1] for line in cache.decode('utf-8').splitlines()
        if line.startswith('CUTLASS_DIR:PATH=')]
    if len(paths)!=1 or not paths[0]:raise ValueError('missing unique CUTLASS layout binding')
    try:
        root=Path(paths[0]).resolve(strict=True)
        folder=root/'include/cutlass/transform/threadblock';observed={}
        for name,wanted in MAINLOOP_LAYOUT_SHA256.items():
            path=folder/name
            if not stat.S_ISREG(path.lstat().st_mode) or path.resolve(strict=True).parent!=folder.resolve(strict=True):
                raise ValueError('mainloop layout header is not an owned regular file')
            raw=path.read_bytes()
            if len(raw)>1024*1024:raise ValueError('mainloop header exceeds bound')
            digest=hashlib.sha256(raw).hexdigest()
            if digest!=wanted:raise ValueError('unsupported private mainloop header layout: '+name)
            observed[name]=digest
        return observed
    except OSError as error:raise ValueError('missing pinned mainloop layout header') from error


def validate_probe_build_binding(source,build):
    source=Path(source).resolve(strict=True);build=Path(build).resolve(strict=True)
    if source==build:raise ValueError('probe build must be out of source')
    cache=(build/'CMakeCache.txt').read_bytes()
    if len(cache)>1024*1024:raise ValueError('probe CMake cache exceeds bound')
    homes=[line.split('=',1)[1] for line in cache.decode('utf-8').splitlines()
        if line.startswith('CMAKE_HOME_DIRECTORY:INTERNAL=')]
    if len(homes)!=1 or Path(homes[0]).resolve(strict=True)!=source:
        raise ValueError('probe CMake source binding mismatch')
    if not (source/'CMakeLists.txt').is_file():raise ValueError('probe source configuration missing')
    return source,build


def build_storage_probes(source,build,log_dir,timeout):
    source,build=validate_probe_build_binding(source,build)
    observe_mainloop_layout(build)
    observe_public_layout_headers(build)
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0<timeout<=3600:
        raise ValueError('invalid compile-only probe build timeout')
    from tools.cube4_production_preflight import run_scorer
    command=['cmake','--build',str(build),'--target',*[v[0] for v in PROBES.values()],'-j2']
    if run_scorer(command,Path(log_dir)/'storage_probe_build.log',timeout)!=0:
        raise RuntimeError('compile-only storage probe build failed')
    return observe_storage_probes(source,build)


def observe_storage_probes(source,build):
    source,build=validate_probe_build_binding(source,build)
    layout=observe_mainloop_layout(build)
    public_layout=observe_public_layout_headers(build)
    tables={};digests={}
    for key,(target,validator) in PROBES.items():
        binary=build/target
        if not stat.S_ISREG(binary.lstat().st_mode) or binary.resolve(strict=True).parent!=build:
            raise ValueError('probe binary is not an owned regular build output')
        digests[target]=hashlib.sha256(binary.read_bytes()).hexdigest()
        result=subprocess.run([str(binary)],capture_output=True,check=True,timeout=5)
        limit=65536 if key.endswith('_public_fields') else 16384
        if len(result.stdout)>limit or result.stderr:raise ValueError('unexpected compile-only probe output')
        table=json.loads(result.stdout,object_pairs_hook=unique)
        validator(table);tables[key]=table
        if key=='gemm_parameter':
            result=subprocess.run([str(binary),'--sm80'],capture_output=True,check=True,timeout=5)
            if len(result.stdout)>16384 or result.stderr:raise ValueError('unexpected SM80 probe output')
            table=json.loads(result.stdout,object_pairs_hook=unique)
            if table.get('scope')!='compiled_SM80_classic_gemm_parameters_not_admission':
                raise ValueError('SM80 probe did not execute requested table mode')
            validate_gemm_parameter_storage(table)
            if hashlib.sha256(binary.read_bytes()).hexdigest()!=digests[target]:
                raise ValueError('GEMM probe changed between architecture observations')
            tables['gemm_parameter_sm80']=table
    sources={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        source/'CMakeLists.txt',source/'tools/stream1_parameter_storage_probe.cu',
        source/'tools/stream1_attention_storage_probe.cu',source/'tools/stream1_gemm_parameter_storage_probe.cu',
        source/'tools/stream1_gemm_iterator_probe.cu',
        source/'tools/stream1_gemm_field_ledger_probe.cu',source/'tools/stream1_attention_field_ledger_probe.cu',
        source/'tools/public_parameter_storage.py',source/'tools/parameter_public_storage_v1.json',
        source/'tools/stream1_gemm_nested_storage_probe.cu',source/'tools/gemm_nested_storage_v1.json',
        source/'tools/graph_argument_coverage.py',source/'tools/network_member_consumption.py',
        source/'tools/graph_network_contract.py',source/'tools/cube4_production_preflight.py',
        source/'tools/storage_probe_bundle.py',
        source/'tools/mainloop_iterator_observation.hpp',source/'tools/graph_gemm_members.hpp',
        source/'tools/graph_gemm_iterator_contract.py',source/'tools/gemm_iterator_values.py',
        source/'cuda/stream1.hpp',source/'cuda/stream1_transformer_layernorm_policy.hpp')}
    return dict(tables=tables,binary_sha256=digests,source_sha256=sources,
        mainloop_layout_sha256=layout,
        public_parameter_layout_sha256=public_layout,
        cmake_cache_sha256=hashlib.sha256((build/'CMakeCache.txt').read_bytes()).hexdigest(),
        production_admitted=False)
