"""Graph-bound export of our GNN, with immutable file receipts."""
import copy
import hashlib
import json
from pathlib import Path
import torch
from .errors import NativeBackendError, NativeUnavailable
from .pancake_gnn import PancakeGNN

def _check_graph(contract, n):
    identity = tuple(range(n))
    moves = {tuple(reversed(identity[:k]))+identity[k:] for k in range(2,n+1)}
    if (contract.state_len != n or contract.center != identity or
            tuple(sorted(contract.start)) != identity or set(contract.generators) != moves or
            contract.move_count != n-1):
        raise NativeUnavailable('PancakeGNN requires identity-centered, distinct-value pancake graph with all prefix flips')

def export_gnn(model, contract, run_dir):
    _check_graph(contract, model.n)
    # Do not erase custom forward/preprocessing hooks or subclasses.
    if type(model) is not PancakeGNN or any(m._forward_hooks or m._forward_pre_hooks for m in model.modules()):
        raise NativeUnavailable('GNN export requires our unchanged PancakeGNN without forward hooks')
    directory = Path(run_dir).resolve()/'weights'
    directory.mkdir(parents=True,exist_ok=False)
    frozen = copy.deepcopy(model).cpu().eval().half()
    for parameter in frozen.parameters():
        if not torch.isfinite(parameter).all():
            raise NativeBackendError('GNN parameters overflowed FP16 or contain nonfinite values')
    for module in frozen.modules():
        if hasattr(module,'use_cutlass'): module.use_cutlass=False
    scripted = torch.jit.script(frozen)
    with (directory/'backbone.pt').open('wb') as artifact:
        torch.jit.save(scripted,artifact)
    product = 1
    for count in model.neighbor_counts:
        product *= model._actions(count,torch.device('cpu')).numel()
    maximum = model.max_frontier_states//product if model.max_frontier_states>0 else 2147483647
    if maximum<1:
        raise NativeUnavailable('GNN frontier cap activates even for one root; stochastic cap is unsupported')
    metadata = dict(schema_version=1,family='pancake_gnn',graph_hash=contract.graph_hash,
                    n=model.n,d_model=model.d_model,max_state_batch=maximum,
                    files={'backbone.pt':hashlib.sha256((directory/'backbone.pt').read_bytes()).hexdigest()})
    manifest = dict(state_len=model.n,num_classes=model.n,hidden1=model.d_model,
                    hidden2=model.d_model,residual_count=1,output_dim=1,dtype='fp16',
                    normalization='none',graph_hash=contract.graph_hash)
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (directory/'gnn.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return validate_gnn(directory,contract)

def validate_gnn(directory, contract):
    from .models import PreparedModel, _load_unambiguous_manifest
    directory=Path(directory).resolve()
    try:
        metadata, raw = _load_unambiguous_manifest(directory/'gnn.json')
        manifest, manifest_raw = _load_unambiguous_manifest(directory/'manifest.json')
        n=metadata['n'];d=metadata['d_model'];batch=metadata['max_state_batch']
        if (type(n) is not int or type(d) is not int or d<=0 or d%8 or
                type(batch) is not int or not 1<=batch<=2147483647 or
                metadata['schema_version']!=1 or metadata['family']!='pancake_gnn' or
                metadata['graph_hash']!=contract.graph_hash or manifest.get('graph_hash')!=contract.graph_hash or
                manifest.get('state_len')!=n or manifest.get('num_classes')!=n or
                manifest.get('hidden1')!=d or manifest.get('hidden2')!=d or
                manifest.get('output_dim')!=1 or manifest.get('dtype')!='fp16'):
            raise NativeBackendError('invalid GNN artifact shape, graph or precision')
        _check_graph(contract,n)
        if set(metadata['files'])!={'backbone.pt'}:
            raise NativeBackendError('unexpected GNN artifact files')
        payload=(directory/'backbone.pt').read_bytes()
        if hashlib.sha256(payload).hexdigest()!=metadata['files']['backbone.pt']:
            raise NativeBackendError('GNN backbone checksum mismatch')
        digest=hashlib.sha256(raw+manifest_raw+payload+contract.graph_hash.encode('ascii')).hexdigest()
        return PreparedModel(directory,dict(manifest,gnn=metadata),'pancake_gnn',digest,
                             ('manifest.json','gnn.json','backbone.pt'))
    except (OSError,KeyError,TypeError,ValueError) as error:
        raise NativeBackendError(f'invalid GNN artifact: {error}') from error
