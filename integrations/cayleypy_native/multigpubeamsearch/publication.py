"""Replay-checked publication through the existing public results client."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from urllib.parse import quote
import uuid

CLIENT_COMMIT = '73937259b360ff9abead62bc90f4cca433114b0a'
CLIENT_SHA256 = '3f206bf21d0314faf4caf7867e637e49423088473814b4ddb1473c0ead62b26f'
PUBLISH_TIMEOUT_SECONDS = 7_200
PUBLISH_SUBMIT_PROCESS_TIMEOUT_SECONDS = 120
PUBLISH_POLL_PROCESS_GRACE_SECONDS = 60


def kaggle_run_url(owner, slug):
    """Return the canonical public Kaggle notebook URL for result provenance."""
    return f'https://www.kaggle.com/code/{quote(str(owner), safe="")}/{quote(str(slug), safe="")}'


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
                        solver_commit, directory='beam_results', client=None,
                        competition=None, puzzle_type=None,
                        model_metadata=None):
    """Create a replay-checked Kaggle result using the catalog's v1 schema.

    Graph identity and move definitions come from the supplied CayleyPy graph.
    A caller may supply a schema-compatible model object for another registered
    model family. The Cube555 blend metadata is inferred for the default example.
    """
    from .contracts import GraphContract
    from .models import NativeEnsemble
    contract = GraphContract.from_graph(graph, start_state)
    paths = result.solutions
    if not paths: return None
    if not isinstance(competition, str) or not competition.strip():
        raise ValueError('competition must be a nonempty catalog identifier')
    if not isinstance(puzzle_type, str) or not puzzle_type.strip():
        raise ValueError('puzzle_type must be a nonempty graph/puzzle label')
    if not all(contract.replay(path) for path in paths): raise ValueError('Publication replay failed')
    m = result.native_metadata
    if len(m.get('devices', ())) != 2:
        raise ValueError('The Kaggle results catalog v1 requires exactly 2 GPUs')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    client = client or client_module(directory/'publisher')
    if model_metadata is None:
        is_cube555_blend = (
            isinstance(predictor, NativeEnsemble)
            and len(predictor.models) == 2
            and tuple(getattr(model, "backend", None) for model in predictor.models)
            == ("cube555_transformer", "cube555_mlp")
        )
        if is_cube555_blend:
            manifests = [json.loads((model.weights_dir/'manifest.json').read_text()) for model in predictor.models]
            if all('checkpoint_sha256' in manifest for manifest in manifests):
                scripts = [hashlib.sha256((model.weights_dir/'features.ts').read_bytes()).hexdigest()
                           for model in predictor.models]
                model_metadata = dict(filename='cube555-native-blend', sha256=m['model_artifact_hash'],
                    format='cube555-q-blend', manifest=dict(state_len=contract.state_len,
                        num_classes=manifests[0]['num_classes'], output_dim=contract.move_count,
                        dtype='fp16', model_family='cube555_q_blend',
                        blend_weights=list(predictor.coefficients),
                        checkpoint_sha256=[v['checkpoint_sha256'] for v in manifests],
                        script_sha256=client._sha(scripts),
                        score_contract='parent_q_fp32_blend_then_clamp_round'))
        if model_metadata is None:
            manifest = m.get('model_manifest', {})
            normalization = manifest.get('normalization')
            formats = {'batchnorm_folded': 'batchnorm-folded', 'layernorm': 'resmlp-layernorm'}
            model_format = formats.get(normalization)
            if not model_format or manifest.get('dtype') != 'fp16' or 'ensemble' in manifest:
                raise ValueError('This model needs explicit model_metadata supported by the CayleyPy catalog schema')
            model_keys = ('state_len', 'num_classes', 'hd1', 'hd2', 'nrd', 'output_dim',
                          'dtype', 'normalization', 'layout', 'batchnorm', 'embeddingbag', 'embedding')
            model_manifest = {key: manifest[key] for key in model_keys if key in manifest}
            model_metadata = dict(filename='native-model', sha256=m['model_artifact_hash'],
                                  format=model_format, manifest=model_manifest)
    if not isinstance(model_metadata, dict):
        raise ValueError('model_metadata must be the model object required by the CayleyPy catalog schema')
    model_manifest = model_metadata.get('manifest')
    if (not isinstance(model_manifest, dict) or type(model_manifest.get('output_dim')) is not int
            or model_manifest['output_dim'] <= 0):
        raise ValueError('model_metadata.manifest.output_dim must describe the selected predictor')
    if (not isinstance(model_metadata.get('filename'), str)
            or not isinstance(model_metadata.get('format'), str)
            or not isinstance(model_metadata.get('sha256'), str)):
        raise ValueError('model_metadata needs filename, format, and sha256 for the catalog')
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
        kaggle=dict(owner=kaggle_owner,slug=kaggle_slug,version=kaggle_version,
            run_url=kaggle_run_url(kaggle_owner,kaggle_slug),notebook_sha256=client._sha(cells)),
        competition=competition, run_id='cayleypy-'+uuid.uuid4().hex, solver_commit=solver_commit,
        profile=dict(requested_beam=requested,effective_beam=width,alignment_delta=width-requested,
            selected_profile=m['profile']['pipeline_calibration']['selection'],evidence='native-auto-admitted',
            profile_evidence_version=1,profile_power=max(1,(width-1).bit_length()),
            model_class='output1' if model_manifest['output_dim'] == 1 else 'output_move_count',world_size=2),
        runtime=runtime, model=model_metadata,
        hardware=dict(platform='kaggle',gpu_names=m['profile']['gpu_names'],accelerator_count=2,world_size=2),
        timings=dict(solve_us=round(m['native_seconds']*1e6),wall_us=round(m['elapsed_seconds']*1e6)))
    context = {str(puzzle_id):dict(puzzle_type=puzzle_type,initial_state=list(contract.start),
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
    """Submit bounded chunks, then poll all receipts against one shared deadline."""
    path = prepare_publication(result,graph,start_state,**kwargs)
    if path is None:
        print('No solutions to publish.'); return None
    client = client_module(path.parent/'publisher')
    envelopes=json.loads(path.read_text())['results']
    receipts=[]
    part_manifests=[]
    parts=[]
    endpoint_origin=None
    deadline=time.monotonic()+PUBLISH_TIMEOUT_SECONDS
    for offset in range(0,len(envelopes),100):
        part=path.with_name(path.stem+f'-part{offset//100}.json')
        part.write_bytes(client.canonical_bytes(dict(schema_version=1,results=envelopes[offset:offset+100])))
        receipt=part.with_suffix('.receipts.json');receipts.append(str(receipt))
        remaining=deadline-time.monotonic()
        if remaining<=0:
            return dict(status='pending_or_failed',payload=str(path),receipts=receipts)
        completed=subprocess.run([sys.executable,client.__file__,'submit',str(part),
            '--manifest',str(receipt)],capture_output=True,text=True,
            timeout=min(PUBLISH_SUBMIT_PROCESS_TIMEOUT_SECONDS,max(1,remaining)))
        part.with_suffix('.publish.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
        if completed.returncode:
            print(f'Publication not confirmed; saved payload {path} and receipts {receipt}')
            return dict(status='pending_or_failed',payload=str(path),receipts=receipts)
        manifest=json.loads(receipt.read_text(encoding='utf-8'))
        current_origin=manifest.get('endpoint_origin')
        if not isinstance(current_origin,str) or (endpoint_origin is not None and current_origin!=endpoint_origin):
            return dict(status='pending_or_failed',payload=str(path),receipts=receipts)
        endpoint_origin=current_origin
        parts.extend(manifest.get('parts',[]))
        receipts_in_part=manifest.get('receipts',[])
        if not isinstance(receipts_in_part,list):
            return dict(status='pending_or_failed',payload=str(path),receipts=receipts)
        part_manifests.append(receipt)
        # Preserve each part receipt immediately; accepted means queued, not published.
        print('Submitted part:',completed.stdout.strip())

    poll_manifest=path.with_name(path.stem+'-poll-manifest.json')
    combined=dict(manifest_version=1,endpoint_origin=endpoint_origin,
        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),parts=parts,receipts=[])
    for manifest_path in part_manifests:
        combined['receipts'].extend(json.loads(manifest_path.read_text(encoding='utf-8'))['receipts'])
    poll_manifest.write_text(json.dumps(combined,indent=2),encoding='utf-8')
    remaining=max(0,int(deadline-time.monotonic()))
    if remaining<=0:
        return dict(status='pending_or_failed',payload=str(path),receipts=[str(p) for p in part_manifests],
                    poll_manifest=str(poll_manifest),unresolved=len(combined['receipts']))
    try:
        completed=subprocess.run([sys.executable,client.__file__,'poll','--manifest',str(poll_manifest),
            '--timeout',str(remaining)],capture_output=True,text=True,
            timeout=remaining+PUBLISH_POLL_PROCESS_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        print(f'Publication polling timed out; saved {poll_manifest}')
        return dict(status='pending_or_failed',payload=str(path),receipts=[str(p) for p in part_manifests],
                    poll_manifest=str(poll_manifest),unresolved=len(combined['receipts']))
    poll_manifest_path=poll_manifest
    combined=json.loads(poll_manifest_path.read_text(encoding='utf-8'))
    states={item.get('submission_id'):item.get('state') for item in combined['receipts']}
    for manifest_path in part_manifests:
        part_receipts=json.loads(manifest_path.read_text(encoding='utf-8'))
        for item in part_receipts['receipts']:
            if item.get('submission_id') in states:
                item['state']=states[item['submission_id']]
        manifest_path.write_text(json.dumps(part_receipts,indent=2),encoding='utf-8')
    published=sum(item.get('state')=='published' for item in combined['receipts'])
    duplicate=sum(item.get('state')=='duplicate' for item in combined['receipts'])
    rejected=sum(item.get('state')=='rejected' for item in combined['receipts'])
    unresolved=len(combined['receipts'])-published-duplicate-rejected
    status='published' if completed.returncode==0 and rejected==0 and unresolved==0 else 'pending_or_failed'
    print(f'Publication poll: {published} published, {duplicate} duplicate, '
          f'{rejected} rejected, {unresolved} unresolved')
    return dict(status=status,payload=str(path),receipts=[str(p) for p in part_manifests],
                poll_manifest=str(poll_manifest_path),published=published,duplicate=duplicate,
                rejected=rejected,unresolved=unresolved)
