"""Measure Stream1 against the very same files used by full-depth calibration."""
import json
from pathlib import Path
from .calibration_session import InferenceSession
from .calibration_report import matched_step_report
from .models import verify_prepared_model


def measure_matched(contract, model, runtime, environment, directory, fixture,
                    selection, micro, *, deadline, devices):
    if model.backend!='ensemble' or runtime.build_metadata.get('calibration_protocol')!='json-session-v1':
        return {'measurement_scope':'unavailable','reason':'matched file reader requires ensemble session probe'}
    verify_prepared_model(model,contract)
    directory=Path(directory);directory.mkdir()
    helper=runtime.runner.parent/runtime.build_metadata['calibration_binary_name']
    sessions=[]
    try:
        for rank,file in enumerate(fixture['files']):
            inputs=directory/f'input-{rank}';inputs.mkdir()
            manifest=dict(model.manifest['ensemble'])
            manifest['models']=[dict(entry,weights_dir=str(model.weights_dir/entry['weights_dir']))
                                for entry in manifest['models']]
            manifest.update(generators=[list(g) for g in contract.generators],
                calibration_frontier_file=file['path'])
            (inputs/'ensemble.json').write_text(json.dumps(manifest))
            sessions.append(InferenceSession([str(helper),str(inputs),str(micro),
                str(file['parents']),str(rank),'--session'],environment,
                directory/f'rank-{rank}.log',deadline=deadline))
        for rank,session in enumerate(sessions):
            ready=session.receive()
            if ready.get('ready') is not True or ready.get('device')!=rank:
                raise RuntimeError('matched Stream1 probe rank is not ready')
        for session,file in zip(sessions,fixture['files']):session.send(micro,file['parents'])
        rows=[session.receive() for session in sessions]
        for rank,row in enumerate(rows):
            if (row.get('correctness_passed') is not True or row.get('numeric_error')!=0 or
                row.get('parents')!=fixture['files'][rank]['parents'] or row.get('device')!=rank or
                row.get('score_input')!='full_frontier_file'):
                raise ValueError('matched probe did not confirm exact file workload and correctness')
    finally:
        for session in sessions:session.close()
    import torch
    identity=dict(parents=fixture['global_parents'],
        gpu_uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in devices],
        frontier_sha256=[file['sha256'] for file in fixture['files']],
        model_sha256=model.artifact_hash,build_sha256=runtime.build_metadata['binary_sha256'],
        precision='fp16/fp32',executor='libtorch_eager',correctness_passed=True)
    repeats=min(len(row['seconds']) for row in rows)
    inference=dict(identity,seconds_by_rank=[[row['seconds'][i] for row in rows] for i in range(repeats)])
    selected=selection['estimate']['profile']
    pipeline=dict(identity,seconds_by_rank=[row['seconds_by_rank'] for row in selection['measurements']
        if row['profile']==selected])
    report=matched_step_report(inference,pipeline)
    (directory/'receipts.json').write_text(json.dumps(dict(inference=inference,pipeline=pipeline,
        report=report,helper_sha256=runtime.build_metadata['calibration_binary_sha256']),indent=2))
    verify_prepared_model(model,contract)
    return report
