"""Replay-checked publication through the existing public results client."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
import uuid

CLIENT_COMMIT = 'f683dc64ca86408e2f3d4b54bb207b8d75ee5e51'
CLIENT_SHA256 = '34959e8e007a27f1d723c5ec5885dae24eb1dbd4295589fa69c0e891ae97f3dd'


def client_module(cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    path = cache / 'cayleypy_submit.py'
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != CLIENT_SHA256:
        url = f'https://raw.githubusercontent.com/TryDotAtwo/cayleypy-beam-results/{CLIENT_COMMIT}/cli/cayleypy_submit.py'
        with urllib.request.urlopen(url, timeout=60) as response: data = response.read(1_000_001)
        if hashlib.sha256(data).hexdigest() != CLIENT_SHA256: raise ValueError('Publisher download verification failed')
        temporary = path.with_suffix('.tmp'); temporary.write_bytes(data); temporary.replace(path)
    spec = importlib.util.spec_from_file_location('_cayleypy_results_client', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def prepare_publication(result, graph, start_state, *, puzzle_id, predictor, author,
                        kaggle_owner, kaggle_slug, kaggle_version, notebook_path,
                        solver_commit, directory='cube555_runs', client=None):
    """Create the established v1 Cube555 envelope. Never send credentials or logs."""
    from .contracts import GraphContract
    from .models import NativeEnsemble
    contract = GraphContract.from_graph(graph, start_state)
    paths = result.solutions
    if not paths: return None
    if not all(contract.replay(path) for path in paths): raise ValueError('Publication replay failed')
    m = result.native_metadata
    if len(m['devices']) != 2:
        raise ValueError('The current Cube555 publication schema requires 2 GPUs; solutions remain saved locally')
    if not isinstance(predictor, NativeEnsemble) or len(predictor.models) != 2:
        raise ValueError('Cube555 publication requires the Transformer/MLP ensemble metadata')
    manifests = [json.loads((model.weights_dir/'manifest.json').read_text()) for model in predictor.models]
    scripts = [hashlib.sha256((model.weights_dir/'features.ts').read_bytes()).hexdigest() for model in predictor.models]
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    client = client or client_module(directory/'publisher')
    notebook = json.loads(Path(notebook_path).read_text(encoding='utf-8'))
    cells = [dict(cell_type=c['cell_type'], source=c['source']) for c in notebook['cells']]
    width = m['effective_beam_width']; requested = m['requested_beam_width']
    if requested == 'max': requested = width
    actual = m['execution_parameters']
    runtime = {key.lower(): actual[key] for key in (
        'B_MICRO','STREAM1_CONCURRENCY','STREAM3_RING_SLOTS','SHARD_COUNT',
        'SHARD_CAPACITY_SCALE_PPM','STREAM4_BATCH_CANDIDATES','STREAM4_TRIGGER_CANDIDATES','STREAM4_ACTIVE_SORT_SLOTS')}
    runtime.update(touch_bfs_radius=m['effective_touch_bfs_radius'], solution_mode=m['solution_mode'],
                   max_depth=m['requested_max_steps'], max_collected_solutions=m['max_collected_solutions'])
    common = dict(author=dict(name=author, verification='claimed', kaggle_username=kaggle_owner),
        kaggle=dict(owner=kaggle_owner,slug=kaggle_slug,version=kaggle_version,notebook_sha256=client._sha(cells)),
        competition='cayley-py-555-cube', run_id='cube555-'+uuid.uuid4().hex, solver_commit=solver_commit,
        profile=dict(requested_beam=requested,effective_beam=width,alignment_delta=width-requested,
            selected_profile=m['profile']['pipeline_calibration']['selection'],evidence='native-auto-admitted',
            profile_evidence_version=1,profile_power=max(1,(width-1).bit_length()),model_class='output_move_count',world_size=2),
        runtime=runtime, model=dict(filename='cube555-native-blend',sha256=m['model_artifact_hash'],format='cube555-q-blend',
            manifest=dict(state_len=150,num_classes=manifests[0]['num_classes'],output_dim=30,dtype='fp16',
                model_family='cube555_q_blend',blend_weights=list(predictor.coefficients),
                checkpoint_sha256=[v['checkpoint_sha256'] for v in manifests],script_sha256=client._sha(scripts),
                score_contract='parent_q_fp32_blend_then_clamp_round')),
        hardware=dict(platform='kaggle',gpu_names=m['profile']['gpu_names'],accelerator_count=2,world_size=2),
        timings=dict(solve_us=round(m['native_seconds']*1e6),wall_us=round(m['elapsed_seconds']*1e6)))
    context = {str(puzzle_id):dict(puzzle_type='cube_5/5/5',initial_state=list(contract.start),
        central_state=list(contract.center), generators=dict(zip(contract.generator_names,map(list,contract.generators))))}
    config = client.PublisherConfig(1,common,context,{})
    details = {tuple(row['path']): row for row in m.get('collected_solutions',[])}
    envelopes=[]
    for i,path in enumerate(paths):
        detail=details.get(tuple(path),{})
        row=dict(puzzle_id=str(puzzle_id),solution='.'.join(contract.generator_names[j] for j in path),
            solved_depth=str(detail.get('solved_depth',len(path))),collection_index=str(i),
            collection_status=m.get('collection_status','first_solution'),
            search_mode={'after_original':'after'}.get(m.get('reflect_mode','off'),m.get('reflect_mode','off')),
            final_orientation=detail.get('final_orientation','original'))
        for field, target in [('searched_path','searched_solution'),('reflected_source_path','reflected_source_solution')]:
            if field in detail: row[target]='.'.join(contract.generator_names[j] for j in detail[field])
        if 'touch_depth' in detail: row['touch_depth']=str(detail['touch_depth'])
        envelopes.append(client.expand_solution_row(config,row))
    path = directory / ('publication-'+common['run_id']+'.json')
    path.write_bytes(client.canonical_bytes(dict(schema_version=1,results=envelopes)))
    client.preflight(path)
    return path


def publish_result(result, graph, start_state, **kwargs):
    """Publish all solutions; retain payload and receipts on network/ingest failure."""
    path = prepare_publication(result,graph,start_state,**kwargs)
    if path is None:
        print('No solutions to publish.'); return None
    client = client_module(path.parent/'publisher')
    envelopes=json.loads(path.read_text())['results']
    receipts=[]
    for offset in range(0,len(envelopes),100):
        part=path.with_name(path.stem+f'-part{offset//100}.json')
        part.write_bytes(client.canonical_bytes(dict(schema_version=1,results=envelopes[offset:offset+100])))
        receipt=part.with_suffix('.receipts.json');receipts.append(str(receipt))
        completed=subprocess.run([sys.executable,client.__file__,'submit',str(part),'--wait',
            '--manifest',str(receipt),'--timeout','300'],capture_output=True,text=True,timeout=360)
        part.with_suffix('.publish.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        if completed.returncode:
            print(f'Publication not confirmed; saved payload {path} and receipts {receipt}')
            return dict(status='pending_or_failed',payload=str(path),receipts=receipts)
        print('Publication confirmed:',completed.stdout.strip())
    return dict(status='published',payload=str(path),receipts=receipts)
