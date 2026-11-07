"""Bound independent intended 1D CUDA API geometry; no candidate-derived defaults."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
from tools.cube4_numeric_gate import unique


def byte_counts(expected):
    outer=expected['outer_microbatch'];inner=expected['transformer_microbatch']
    if type(outer) is not int or type(inner) is not int or not 0<inner<=outer<=65536:
        raise ValueError('invalid independent transfer batching')
    return sorted({48*inner} | ({48*(outer%inner)} if outer%inner else set()))


def validate_geometry_reference(value,expected):
    keys={'schema_version','scope','production_admitted','runtime_version','device','cases','memset_reference'}
    if (not isinstance(value,dict) or set(value)!=keys or type(value['schema_version']) is not int or
        value['schema_version']!=1 or value['production_admitted'] is not False or
        value['scope']!='independent_cuda_1d_transfer_geometry_not_admission' or
        type(value['runtime_version']) is not int or value['runtime_version']!=12080):
        raise ValueError('unsupported independent transfer reference scope/runtime')
    device={key:expected['device'][key] for key in ('device','sm','uuid_hex')}
    def frozen(item): return json.dumps(item,sort_keys=True,allow_nan=False)
    if frozen(value['device'])!=frozen(device):raise ValueError('transfer reference device mismatch')
    wanted=[]
    for size in byte_counts(expected):
        geometry=dict(bytes=size,source_pitched=dict(pitch=0,xsize=0,ysize=0),
                                destination_pitched=dict(pitch=0,xsize=0,ysize=0),copy_geometry=copy_geometry(size))
        wanted.append(dict(bytes=size,add_node=geometry,capture_async=geometry))
    if frozen(value['cases'])!=frozen(wanted):
        raise ValueError('independent public 1D API reference geometry mismatch')
    reset=dict(pitch=0,width=4096,height=1,element_size=1,value=0)
    if frozen(value['memset_reference'])!=frozen(dict(add_node=reset,capture_async=reset)):
        raise ValueError('independent public memset API reference geometry mismatch')
    return value

def copy_geometry(size):
    return dict(source_array_null=True,destination_array_null=True,
        source_position=dict(x=0,y=0,z=0),destination_position=dict(x=0,y=0,z=0),
        extent=dict(width=size,height=1,depth=1))


def build_geometry_probe(source,build,output,timeout):
    from tools.storage_probe_bundle import validate_probe_build_binding
    validate_probe_build_binding(source,build)
    with (Path(output)/'transfer_geometry_probe_build.log').open('xb') as log:
        subprocess.run(['cmake','--build',str(build),'--target','stream1_transfer_geometry_probe','-j2'],
            stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def observe_geometry_probe(source,build,expected):
    source=Path(source);binary=Path(build)/'stream1_transfer_geometry_probe'
    if os.name!='posix':raise ValueError('bounded transfer observation requires POSIX')
    import resource
    if not stat.S_ISREG(binary.lstat().st_mode):raise ValueError('transfer probe must be regular')
    fingerprint=hashlib.sha256(binary.read_bytes()).hexdigest()
    def cap(): resource.setrlimit(resource.RLIMIT_FSIZE,(65536,65536))
    with tempfile.TemporaryFile() as output:
        subprocess.run([str(binary),str(expected['device']['device']),
                        *map(str,byte_counts(expected))],stdout=output,stderr=subprocess.DEVNULL,
            check=True,timeout=15,preexec_fn=cap)
        if output.tell()>65536:raise ValueError('transfer reference output exceeds bound')
        output.seek(0);value=json.loads(output.read(),object_pairs_hook=unique)
    validate_geometry_reference(value,expected)
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=fingerprint:
        raise ValueError('transfer probe changed during observation')
    return dict(table=value,binary_sha256=fingerprint,source_sha256={name:
        hashlib.sha256((source/name).read_bytes()).hexdigest() for name in (
            'tools/stream1_transfer_geometry_probe.cu','tools/transfer_geometry_observation.py',
            'tools/graph_transfer_inventory.hpp','tools/graph_transfer_contract.py','CMakeLists.txt')})


def validate_observed_pitched_geometry(value,reference,expected):
    if not isinstance(reference,dict) or set(reference)!={'table','binary_sha256','source_sha256'}:
        raise ValueError('missing independent pitched geometry reference')
    validate_geometry_reference(reference['table'],expected)
    wanted=dict(pitch=0,xsize=0,ysize=0)
    for lane in value['lanes']:
        for operation in lane['operations']:
            if operation['kind']=='memset':
                geometry=operation.get('memset_geometry')
                wanted_reset=dict(pitch=0,width=operation['bytes'],height=1)
                if (json.dumps(geometry,sort_keys=True,allow_nan=False)!=json.dumps(wanted_reset,sort_keys=True) or
                    type(operation['element_size']) is not int or operation['element_size']!=1):
                    raise ValueError('captured memset geometry mismatch')
                continue
            if operation['kind']!='memcpy':raise ValueError('unknown geometry operation')
            if json.dumps(operation.get('copy_geometry'),sort_keys=True,allow_nan=False)!=json.dumps(copy_geometry(operation['bytes']),sort_keys=True):
                raise ValueError('captured memcpy complete geometry mismatch')
            for endpoint in ('source_pitched','destination_pitched'):
                observed=operation.get(endpoint)
                if (not isinstance(observed,dict) or set(observed)!=set(wanted) or
                    any(type(observed[key]) is not int or observed[key]!=0 for key in wanted)):
                    raise ValueError('captured memcpy pitched geometry mismatch: '+endpoint)
