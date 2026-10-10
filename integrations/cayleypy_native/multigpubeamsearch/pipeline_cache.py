"""Exact-workload component receipts; cached plans never replace admission."""
from __future__ import annotations
import hashlib
import json
import math
import uuid
from pathlib import Path

from .autotune import valid_cached_profile
from .pipeline_profiles import TUNABLE_KEYS, validate_rank_plans

KEYS=TUNABLE_KEYS|{'BEAM_ENSEMBLE_INFERENCE_MICRO'}


def cache_identity(inference, environment, *, beam, world, full_frontier):
    signature=inference.get('signature',{})
    if (signature.get('requested_beam_width')!=beam or signature.get('world_size')!=world
            or not valid_cached_profile(inference,signature,signature.get('max_batch',65536))):
        return None
    return dict(schema=1,policy='exact-component-service-v1',signature=signature,
        parent_batch=inference['parent_batch'],reserve_bytes=inference['reserve_bytes'],
        full_frontier=bool(full_frontier),
        policy_environment={k:v for k,v in environment.items()
            if (k.startswith('BEAM_') or k.startswith('NCCL_'))
            and k not in {'BEAM_NCCL_RUN_ID','BEAM_PUZZLE_INFO_JSON','BEAM_TEST_CSV'}
            and not k.endswith(('_PATH','_FILE','_DIR'))})


def cache_path(root,identity):
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return Path(root)/'pipeline-profiles'/f'{key}.json'


def valid_receipt(data,identity):
    """Reject incomplete cohorts and arbitrary environment injection."""
    try:
        if data['cache_identity']!=identity or data['phase']!='component_calibrated':return False
        if data.get('pipeline_verified') is not False:return False
        env=data['environment'];micro=identity['parent_batch'];world=identity['signature']['world_size']
        if set(env)-KEYS or env.get('BEAM_ENSEMBLE_INFERENCE_MICRO')!=str(micro):return False
        if any(type(v) is not str or not v.isascii() or not v.isdecimal()
               or not 0<int(v)<=2**40 for v in env.values()):return False
        chosen=[r for r in data['tested'] if r['name']==data['selection']]
        if len(chosen)!=1 or chosen[0]['environment']!=env:return False
        plans=chosen[0]['plans'];validate_rank_plans(plans)
        if len(plans)!=world:return False
        if data['requested_beam_width']!=identity['signature']['requested_beam_width']:return False
        if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']!=data['requested_beam_effective']:return False
        if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']<data['requested_beam_width']:return False
        rows=chosen[0]['rank_measurements']
        if len(rows)!=world or {r['rank'] for r in rows}!=set(range(world)):return False
        for row in rows:
            if row.get('correctness_passed') is not True or row.get('full_step_verified') is not False:return False
            if row['shard_capacity']!=plans[row['rank']]['SHARD_CAPACITY_CANDIDATES']:return False
            for key in ('stream3_seconds','stream4_group_seconds','union_seconds'):
                if len(row[key])<5 or any(not math.isfinite(v) or v<=0 for v in row[key]):return False
            for transfer in row['transport']:
                if type(transfer['items']) is not int or transfer['items']<=0:return False
                if len(transfer['seconds'])<5 or any(not math.isfinite(v) or v<=0 for v in transfer['seconds']):return False
        return True
    except (KeyError,ValueError,TypeError,OverflowError):return False


def read_profile(path,identity,admit):
    try:
        data=json.loads(path.read_text())
        if not valid_receipt(data,identity):return None
        plans=admit(data['environment'])  # Fresh all-rank VRAM admission is mandatory.
        validate_rank_plans(plans)
        if plans[0]['GLOBAL_BEAM_WIDTH_EFFECTIVE']!=data['requested_beam_effective']:return None
        if plans[0]['B_MICRO']!=int(data['environment']['BEAM_B_MICRO']):return None
        old=next(r['plans'] for r in data['tested'] if r['name']==data['selection'])
        geometry=('GLOBAL_BEAM_WIDTH_EFFECTIVE','frontier_state_capacity','B_MICRO',
            'WORLD_SIZE','SHARD_COUNT','SHARD_CAPACITY_CANDIDATES','STREAM3_RING_SLOTS','RING_COUNT',
            'STREAM4_ACTIVE_SORT_SLOTS','STREAM4_BATCH_CANDIDATES','STREAM4_BATCH_ALIGNMENT')
        if any(a[k]!=b[k] for a,b in zip(old,plans) for k in geometry):return None
        return dict(data,cache_hit=True,current_rank_plans=plans,
            measurement_scope='cached_exact_capacities_with_fresh_admission',pipeline_verified=False)
    except (OSError,ValueError,KeyError,TypeError):return None


def write_profile(path,identity,data):
    record=dict(data,cache_identity=identity)
    if not valid_receipt(record,identity):return
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(record,indent=2))
    temporary.replace(path)
