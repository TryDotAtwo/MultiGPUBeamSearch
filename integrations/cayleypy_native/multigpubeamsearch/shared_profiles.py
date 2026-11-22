"""Read source-pinned GitHub profile candidates, never forge local receipts.

The source checkout supplies profiles/index.json and hash-addressed JSON data.
A match is a proposal requiring fresh target validation, not a local cache hit.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from .errors import NativeUnavailable

SCHEMA = 1
MAX_BYTES = 2 << 20


def compatibility(contract, model, runtime, devices, beam):
    import torch
    from .profile_identity import portable_model_key
    hardware=[]
    for device in devices:
        p=torch.cuda.get_device_properties(device)
        hardware.append(dict(name=p.name,compute=[p.major,p.minor],
            sm_count=p.multi_processor_count,total_memory=p.total_memory))
    identity = dict(schema=SCHEMA,graph=contract.graph_hash,models=portable_model_key(model),
        native_source=runtime.build_metadata['source_digest'],
        cutlass_source=runtime.build_metadata.get('cutlass_digest'),
        build_mode=runtime.build_metadata.get('build_mode'),
        cuda_architectures=runtime.build_metadata.get('cuda_architectures'),
        shape=runtime.build_metadata['shape'],backend=runtime.build_metadata['inference_backend'],
        torch=torch.__version__,cuda=torch.version.cuda,precision='fp16/fp32',
        hardware=hardware,world_size=len(devices),beam=beam)
    index_sort = runtime.build_metadata.get('experimental_index_sort', False)
    if type(index_sort) is not bool:
        raise ValueError('invalid index sort build identity')
    if index_sort:
        identity['experimental_index_sort'] = True
    return identity


def _json(path):
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('shared profile exceeds size limit')
    return json.loads(path.read_text(encoding='utf-8'))


def _find_profile_python(source_dir, identity):
    """Compatibility reader for wheels without the Rust registry method."""
    root=(Path(source_dir)/'profiles').resolve()
    try:
        index=_json(root/'index.json')
        if index.get('schema') != SCHEMA:return None
        entries=index['profiles']
        if not isinstance(entries,list) or len(entries)>1024:return None
        for entry in entries:
            if entry.get('compatibility') != identity:continue
            digest=entry['sha256']
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):continue
            path=root/'records'/(digest+'.json')
            if path.is_symlink() or not path.resolve().is_relative_to(root) or path.stat().st_size>MAX_BYTES:continue
            payload=path.read_bytes()
            if len(payload)>MAX_BYTES or hashlib.sha256(payload).hexdigest()!=digest:continue
            data=json.loads(payload)
            if data.get('schema')!=SCHEMA or data.get('compatibility')!=identity:continue
            if data.get('evidence',{}).get('status')!='verified':continue
            if not isinstance(data['evidence'].get('source_commit'),str):continue
            batch=data.get('inference',{}).get('parent_batch')
            if type(batch) is not int or not 1<=batch<=65536:continue
            return dict(data,registry_sha256=digest)
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return None
    return None


def find_profile(source_dir, identity):
    """Use the Rust registry reader, with an old-wheel Python fallback."""
    if source_dir is None:
        return None
    try:
        from .rust_control import extension
        native = extension()
    except (ImportError, NativeUnavailable):
        native = None
    reader = getattr(native, 'find_shared_profile', None)
    if callable(reader):
        try:
            encoded = reader(
                str(Path(source_dir) / 'profiles'),
                json.dumps(identity, sort_keys=True, allow_nan=False),
            )
            if encoded is None:
                return None
            if not isinstance(encoded, str):
                return None
            value = json.loads(encoded)
            return value if isinstance(value, dict) else None
        except (OSError, RuntimeError, TypeError, ValueError, OverflowError):
            return None
    return _find_profile_python(source_dir, identity)


def lookup(contract, model, runtime, options, devices, beam):
    """An absent/old source bundle cannot disable normal local calibration."""
    source = getattr(options, 'source_dir', None)
    if source is None:
        return None
    try:
        return find_profile(source, compatibility(contract, model, runtime, devices, beam))
    except (OSError, KeyError, TypeError, AttributeError, ValueError):
        return None


def pipeline_environment(profile, parent_batch):
    from .pipeline_cache import KEYS
    try:
        if profile['inference']['parent_batch'] != parent_batch:
            return None
        env = profile['pipeline']['environment']
        if not isinstance(env, dict) or not env or set(env) - KEYS:
            return None
        if any(not isinstance(v, str) or not v.isascii() or not v.isdecimal() or int(v) <= 0
               for v in env.values()):
            return None
        if env.get('BEAM_ENSEMBLE_INFERENCE_MICRO') != str(parent_batch):
            return None
        return dict(env)
    except (KeyError, TypeError, ValueError):
        return None
