"""Portable profile identity, separate from byte-exact artifact integrity."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile


def _constant_order(source):
    # TorchScript emits literal Final declarations from an unordered set. Only
    # reorder contiguous, side-effect-free literal declarations, never methods
    # or executable expressions. Preserve all other source bytes.
    def ordered(block):
        names=[line.split(":",1)[0].strip() for line in block]
        return sorted(block) if len(set(names))==len(names) and all(x.endswith("\n") for x in block) else block
    result=[];block=[]
    for line in source.splitlines(keepends=True):
        match=re.fullmatch(r"  [A-Za-z_]\w* : Final\[[A-Za-z0-9_., \[\]]+\] = (.*)\n?",line)
        pure=False
        if match:
            try:
                ast.literal_eval(match.group(1))
                pure=True
            except (ValueError,SyntaxError):pass
        if pure:block.append(line)
        else:
            result.extend(ordered(block));block=[];result.append(line)
    result.extend(ordered(block))
    return ''.join(result)


def torchscript_profile_digest(path):
    """Hash executable code/constants/tensors; omit source maps and archive ID."""
    entries={}
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            if item.is_dir():continue
            parts=item.filename.split('/')
            if len(parts)<2 or any(x in ('','..','.') for x in parts):
                raise ValueError('invalid TorchScript archive path')
            name='/'.join(parts[1:])
            if name in entries:raise ValueError('duplicate TorchScript archive member')
            raw=archive.read(item)
            if name=='.data/serialization_id' or (name.startswith('code/') and name.endswith('.debug_pkl')):
                continue
            if name.startswith('code/') and name.endswith('.py'):
                raw=_constant_order(raw.decode('utf-8')).encode('utf-8')
            entries[name]=hashlib.sha256(raw).hexdigest()
    if not entries or 'data.pkl' not in entries:
        raise ValueError('missing TorchScript executable data')
    return hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()


def portable_model_key(model):
    """Cube555 exports retain identical execution content across source paths.

    Byte hashes remain authoritative for loading and local caches. This key only
    selects a proposal which must pass fresh numeric/timing and memory checks.
    Other families retain their original strict artifact identity.
    """
    if model.backend != 'ensemble':return model.artifact_hash
    ensemble=model.manifest['ensemble']
    if any(x['family'] not in ('cube555_transformer','cube555_mlp') for x in ensemble['models']):
        return model.artifact_hash
    members=[]
    for member in ensemble['models']:
        root=model.weights_dir/member['weights_dir']
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        manifest['files']['features.ts']=torchscript_profile_digest(root/'features.ts')
        member_key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        members.append(dict(member,artifact_hash=member_key))
    stable=dict(ensemble,models=members)
    return 'cube555-ts-v1:'+hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()
