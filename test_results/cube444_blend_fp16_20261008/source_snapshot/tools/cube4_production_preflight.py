"""Actual-runner candidate inspection; never promote incomplete admission."""
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.full_score_comparison import compare_production_raw_jobs
import math
import os
import signal
import subprocess
import json
import stat
import hashlib
from tools.cube4_numeric_gate import unique
from tools.cube4_run_supervisor import stop_group
from tools.graph_inventory_contract import validate_graph_inventory_structure


def collect_fresh_candidate(command, run_dir, timeout, observe_inputs, reference, *, _replay_saved=False):
    # Trusted caller independently observes runner/model/reference/device/profile.
    # This transaction cannot admit production; it only publishes fresh numeric
    # candidate evidence. Existing output directories are never reusable input.
    def freeze(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    before_json=freeze(observe_inputs())
    before=json.loads(before_json)
    if (not isinstance(before,dict) or not isinstance(before.get('expected'),dict) or
            not isinstance(before.get('reference_dir'),str) or not before['reference_dir']):
        raise ValueError('missing independently observed candidate bindings')
    reference=json.loads(freeze(reference))
    if freeze(before.get('reference_scores'))!=freeze(reference):
        raise ValueError('comparison reference differs from independently observed input')
    if 'expected_resolved' in before:
        from tools.resolved_execution_contract import canonical_resolved_recipe
        if freeze(before['expected_resolved']) != freeze(canonical_resolved_recipe(before['expected'])):
            raise ValueError('incomplete or contradictory independent resolved expectations')
    # Reject contradictory requested coverage before any scorer process/work.
    if 'compiled_probe_bundle' in before:
        from tools.public_parameter_storage import validate_gemm_public_storage,validate_attention_public_storage,validate_nested_gemm_storage
        tables=before['compiled_probe_bundle']['tables']
        validate_gemm_public_storage(tables.get('gemm_public_fields'))
        validate_attention_public_storage(tables.get('attention_public_fields'))
        validate_nested_gemm_storage(tables.get('gemm_nested_storage'))
    if 'expected_empty_elementwise_arguments' in before:
        from tools.elementwise_arguments_observation import validate_empty_arguments_table
        exclusion=before['expected_empty_elementwise_arguments']
        if (before.get('require_graph_inventory') is not True or not isinstance(exclusion,dict) or
            set(exclusion)!={'table','binary_sha256','cutlass_header_sha256','source_sha256'}):
            raise ValueError('invalid independent empty elementwise exclusion')
        validate_empty_arguments_table(exclusion['table'])
    if 'expected_graph_transfer_geometry' in before:
        from tools.transfer_geometry_observation import validate_geometry_reference
        geometry=before['expected_graph_transfer_geometry']
        if (before.get('require_graph_inventory') is not True or
            before.get('require_graph_transfers') is not True or not isinstance(geometry,dict) or
            set(geometry)!={'table','binary_sha256','source_sha256'}):
            raise ValueError('independent transfer geometry requires captured transfers')
        validate_geometry_reference(geometry['table'],before['expected'])
    if 'expected_graph_gemm_launch_profile' in before:
        from tools.graph_gemm_iterator_contract import validate_gemm_mainloop_geometry
        wanted=before['expected_graph_gemm_launch_profile']
        if (before.get('require_graph_inventory') is not True or not isinstance(wanted,dict) or
                set(wanted)!={'geometry','padded_seq_len'} or type(wanted['padded_seq_len']) is not int or
                wanted['padded_seq_len'] not in (57,64)):
            raise ValueError('invalid independent GEMM launch profile')
        validate_gemm_mainloop_geometry(wanted['geometry'])
    if 'expected_graph_gemm_mainloop_geometry' in before:
        if before.get('require_graph_inventory') is not True:
            raise ValueError('GEMM mainloop values require captured graph inventory')
        from tools.graph_gemm_iterator_contract import validate_gemm_mainloop_geometry
        validate_gemm_mainloop_geometry(before['expected_graph_gemm_mainloop_geometry'])
    if 'expected_graph_gemm_iterator_geometry' in before:
        if before.get('require_graph_inventory') is not True:
            raise ValueError('GEMM iterator values require captured graph inventory')
        from tools.graph_gemm_iterator_contract import validate_gemm_iterator_geometry
        validate_gemm_iterator_geometry(before['expected_graph_gemm_iterator_geometry'])
    if 'expected_graph_attention_members_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('attention members require graph inventory')
    if 'expected_graph_network_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('consumed network members require captured graph inventory')
    if 'expected_graph_dims_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('embedded dimensions require captured graph inventory')
    if 'expected_graph_pointer_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('pointer values require captured graph inventory')
    if 'expected_graph_non_gemm_scalar_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('scalar values require captured graph inventory')
    if before.get('require_graph_dependencies') is True and before.get('require_graph_transfers') is not True:
        raise ValueError('graph dependencies require captured transfer arguments')
    if before.get('require_graph_transfers') is True and before.get('require_graph_inventory') is not True:
        raise ValueError('transfer arguments require captured graph inventory')
    if 'expected_graph_gemm_parameter_storage' in before and 'expected_graph_parameter_layout' not in before:
        raise ValueError('GEMM parameter storage requires parameter layout expectations')
    if 'expected_graph_attention_parameter_storage' in before and 'expected_graph_parameter_layout' not in before:
        raise ValueError('attention parameter storage requires parameter layout expectations')
    if 'expected_graph_parameter_layout' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('parameter layout requires captured graph inventory')
    if 'expected_graph_non_gemm_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('non-GEMM expectations require captured graph inventory')
    if 'expected_graph_attention_profile' in before and before.get('require_graph_inventory') is not True:
        raise ValueError('attention expectations require captured graph inventory')
    scorer_env=None
    if 'expected_cupti_replay' in before:
        from tools.cupti_replay_trace import observe_injection_library
        cupti=before['expected_cupti_replay']
        if (before.get('require_graph_inventory') is not True or
                before.get('require_graph_transfers') is not True or
                not isinstance(cupti,dict) or set(cupti)!={'path','sha256','parser_sha256'} or
                observe_injection_library(cupti['path'])!=cupti):
            raise ValueError('invalid independent CUPTI library binding')
        if os.environ.get('CUDA_INJECTION64_PATH'):
            raise ValueError('ambient injection conflicts with scorer-only CUPTI observation')
        scorer_env=dict(os.environ,CUDA_INJECTION64_PATH=cupti['path'])
    output=run_dir/'output'
    if not _replay_saved:
        run_dir.mkdir(exist_ok=False)
        code=run_scorer([*command,'--validate-stream1-scores',before['reference_dir'],str(output)],
                        run_dir/'runner.log',timeout,**({'env':scorer_env} if scorer_env is not None else {}))
        if code!=0:
            raise RuntimeError('fresh production scorer failed with exit code '+str(code))
    try:
        if not stat.S_ISDIR(output.lstat().st_mode):
            raise ValueError('fresh output must be a real owned directory, not a symlink')
    except OSError as error:
        raise ValueError('fresh scorer output directory missing') from error

    content_hashes={}
    def read_output(name,limit):
        path=output/name
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError('candidate output is not a regular file')
            with path.open('rb') as stream:
                raw=stream.read(limit+1)
            if len(raw)>limit: raise ValueError('candidate output exceeds size bound')
            content_hashes[name]=hashlib.sha256(raw).hexdigest()
            value=json.loads(raw,object_pairs_hook=unique)
            if not isinstance(value,dict): raise ValueError('candidate output must be an object')
            return value
        except (OSError,json.JSONDecodeError) as error:
            raise ValueError('missing or malformed fresh candidate output: '+name) from error

    execution=read_output('execution.json',8*1024*1024)
    scores=read_output('raw_scores.json',256*1024*1024)
    graph_inventory_checked=False
    graph_gemm_checked=False
    graph_attention_checked=False
    graph_non_gemm_checked=False
    graph_parameter_layout_checked=False
    graph_attention_parameter_layout_checked=False
    graph_gemm_parameter_layout_checked=False
    graph_transfers_checked=False
    graph_transfer_pitched_geometry_checked=False
    graph_dependencies_checked=False
    graph_scalars_checked=False
    graph_pointers_checked=False
    graph_dims_checked=False
    graph_network_checked=False
    graph_attention_members_checked=False
    graph_gemm_public_iterators_checked=False
    graph_gemm_mainloop_iterators_checked=False
    graph_gemm_launch_values_checked=False
    argument_coverage=None
    if before.get('require_graph_inventory') is True:
        inventory=read_output('graph_kernel_inventory.json',32*1024*1024)
        if (type(inventory.get('schema_version')) is not int or inventory['schema_version']!=1 or
                inventory.get('scope')!='captured_graph_kernel_inventory_not_admission' or
                inventory.get('production_admitted') is not False or
                inventory.get('kernel_coverage_complete') is not False):
            raise ValueError('graph inventory scope mismatch')
        if freeze(inventory.get('device'))!=freeze(before['expected'].get('device')):
            raise ValueError('graph inventory device mismatch')
        if inventory.get('execution_sha256')!=content_hashes['execution.json']:
            raise ValueError('graph inventory execution content mismatch')
        if before['expected'].get('executor')!='native_cuda_graph':
            raise ValueError('captured graph cannot attest eager execution')
        validate_graph_inventory_structure(inventory,
            lanes=before['expected']['lanes'],sm=before['expected']['device']['sm'])
        if 'expected_graph_gemm_iterator_geometry' in before or 'expected_graph_gemm_mainloop_geometry' in before:
            from tools.graph_gemm_iterator_contract import validate_graph_gemm_public_iterators,validate_graph_gemm_mainloop_iterators
            members=read_output('graph_gemm_members.json',32*1024*1024)
            if (freeze(members.get('device'))!=freeze(before['expected'].get('device')) or
                members.get('execution_sha256')!=content_hashes['execution.json'] or
                members.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']):
                raise ValueError('GEMM iterator member binding mismatch')
            if 'expected_graph_gemm_iterator_geometry' in before:
                validate_graph_gemm_public_iterators(members,inventory,before['expected'],
                    before['expected_graph_gemm_iterator_geometry'])
            if 'expected_graph_gemm_mainloop_geometry' in before:
                validate_graph_gemm_mainloop_iterators(members,inventory,before['expected'],
                    before['expected_graph_gemm_mainloop_geometry'])
                graph_gemm_mainloop_iterators_checked=True
            graph_gemm_public_iterators_checked=True
        if 'expected_graph_gemm_launch_profile' in before:
            from tools.graph_gemm_launch_contract import validate_graph_gemm_launch_values
            members=read_output('graph_gemm_members.json',32*1024*1024)
            if (freeze(members.get('device'))!=freeze(before['expected'].get('device')) or
                    members.get('execution_sha256')!=content_hashes['execution.json'] or
                    members.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']):
                raise ValueError('GEMM launch member binding mismatch')
            wanted=before['expected_graph_gemm_launch_profile']
            validate_graph_gemm_launch_values(members,inventory,before['expected'],
                wanted['geometry'],wanted['padded_seq_len'])
            graph_gemm_launch_values_checked=True
            covered_gemm_members=members
        if 'expected_graph_attention_members_profile' in before:
            from tools.graph_attention_members_contract import validate_graph_attention_members
            wanted=before['expected_graph_attention_members_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len'}:
                raise ValueError('missing independent attention member profile')
            members=read_output('graph_attention_members.json',32*1024*1024)
            if freeze(members.get('device'))!=freeze(before['expected'].get('device')) or members.get('execution_sha256')!=content_hashes['execution.json'] or members.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('attention member binding mismatch')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_attention_members(members,inventory,wanted['model'],profile)
            graph_attention_members_checked=True
            covered_attention_members=members
        if 'expected_graph_network_profile' in before:
            from tools.graph_network_contract import validate_graph_network_members
            wanted=before['expected_graph_network_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','ln_shared_bytes'}:
                raise ValueError('missing independent consumed network profile')
            network=read_output('graph_network_members.json',32*1024*1024)
            if freeze(network.get('device'))!=freeze(before['expected'].get('device')):
                raise ValueError('consumed network device mismatch')
            if network.get('execution_sha256')!=content_hashes['execution.json']:
                raise ValueError('consumed network execution mismatch')
            if network.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('consumed network inventory mismatch')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_network_members(network,inventory,wanted['model'],profile,wanted['ln_shared_bytes'])
            graph_network_checked=True
        if 'expected_graph_dims_profile' in before:
            from tools.graph_dims_contract import validate_graph_embedded_dims
            wanted=before['expected_graph_dims_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','ln_shared_bytes'}:
                raise ValueError('missing independent dimension profile')
            dims_values=read_output('graph_embedded_dims.json',32*1024*1024)
            if freeze(dims_values.get('device'))!=freeze(before['expected'].get('device')):
                raise ValueError('dimension device mismatch')
            if dims_values.get('execution_sha256')!=content_hashes['execution.json']:
                raise ValueError('dimension execution mismatch')
            if dims_values.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('dimension inventory mismatch')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_embedded_dims(dims_values,inventory,wanted['model'],profile,wanted['ln_shared_bytes'])
            graph_dims_checked=True
        if 'expected_graph_pointer_profile' in before:
            from tools.graph_pointer_contract import validate_graph_pointer_roles
            wanted=before['expected_graph_pointer_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','ln_shared_bytes'}:
                raise ValueError('missing independent pointer profile expectations')
            pointer_values=read_output('graph_pointer_roles.json',32*1024*1024)
            if freeze(pointer_values.get('device'))!=freeze(before['expected'].get('device')):
                raise ValueError('pointer device mismatch')
            if pointer_values.get('execution_sha256')!=content_hashes['execution.json']:
                raise ValueError('pointer execution hash mismatch')
            if pointer_values.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('pointer inventory hash mismatch')
            lanes=pointer_values.get('lanes')
            if not isinstance(lanes,list):raise ValueError('missing pointer lane observations')
            captures={lane['lane']:lane for lane in inventory['lanes']}
            for lane in lanes:
                if not isinstance(lane,dict) or type(lane.get('lane')) is not int or lane['lane'] not in captures:
                    raise ValueError('invalid pointer lane binding')
                nodes=lane.get('kernels');native=captures[lane['lane']]['kernels']
                if not isinstance(nodes,list) or len(nodes)!=len(native):raise ValueError('pointer node coverage mismatch')
                for node in nodes:
                    if not isinstance(node,dict) or type(node.get('kernel_index')) is not int or not 0<=node['kernel_index']<len(native):
                        raise ValueError('invalid pointer kernel binding')
                    symbol=inventory['functions'][native[node['kernel_index']]['function_id']]['mangled_name']
                    if node.get('mangled_name')!=symbol:raise ValueError('pointer function identity mismatch')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_pointer_roles(pointer_values,wanted['model'],profile,wanted['ln_shared_bytes'],
                kernel_grids={lid:[node['grid'] for node in capture['kernels']]
                              for lid,capture in captures.items()})
            graph_pointers_checked=True
        if 'expected_graph_non_gemm_scalar_profile' in before:
            from tools.graph_scalar_profile import validate_graph_non_gemm_scalars
            wanted=before['expected_graph_non_gemm_scalar_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','ln_shared_bytes'}:
                raise ValueError('missing independent scalar profile expectations')
            scalar_values=read_output('graph_u32_values.json',32*1024*1024)
            if freeze(scalar_values.get('device'))!=freeze(before['expected'].get('device')):
                raise ValueError('scalar device mismatch')
            if scalar_values.get('execution_sha256')!=content_hashes['execution.json']:
                raise ValueError('scalar execution hash mismatch')
            if scalar_values.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('scalar inventory hash mismatch')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_non_gemm_scalars(scalar_values,inventory,wanted['model'],profile,wanted['ln_shared_bytes'])
            graph_scalars_checked=True
        if before.get('require_graph_transfers') is True:
            from tools.graph_transfer_contract import validate_bound_score_graph_transfers
            transfers=read_output('graph_transfers.json',32*1024*1024)
            validate_bound_score_graph_transfers(transfers,before['expected'],content_hashes)
            graph_transfers_checked=True
            if 'expected_graph_transfer_geometry' in before:
                from tools.transfer_geometry_observation import validate_observed_pitched_geometry
                validate_observed_pitched_geometry(transfers,before['expected_graph_transfer_geometry'],before['expected'])
                graph_transfer_pitched_geometry_checked=True
            if before.get('require_graph_dependencies') is True:
                from tools.graph_transfer_contract import validate_score_graph_dependencies
                validate_score_graph_dependencies(transfers,
                    outer=before['expected']['outer_microbatch'],
                    inner=before['expected']['transformer_microbatch'],lanes=before['expected']['lanes'])
                graph_dependencies_checked=True
        if 'expected_graph_gemm_profile' in before:
            from tools.graph_gemm_profile import validate_graph_gemm_profile
            wanted=before['expected_graph_gemm_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len'}:
                raise ValueError('missing independent GEMM profile expectations')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_gemm_profile(inventory,wanted['model'],profile)
            graph_gemm_checked=True
        if 'expected_graph_attention_profile' in before:
            from tools.graph_attention_profile import validate_graph_attention_profile
            wanted=before['expected_graph_attention_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','storage'}:
                raise ValueError('missing independent attention profile expectations')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_attention_profile(inventory,wanted['model'],profile,wanted['storage'])
            graph_attention_checked=True
        if 'expected_graph_non_gemm_profile' in before:
            from tools.graph_non_gemm_profile import validate_graph_non_gemm_profile
            wanted=before['expected_graph_non_gemm_profile']
            if not isinstance(wanted,dict) or set(wanted)!={'model','padded_seq_len','ln_shared_bytes'}:
                raise ValueError('missing independent non-GEMM profile expectations')
            profile=dict(before['expected'],expected_padded_seq_len=wanted['padded_seq_len'])
            validate_graph_non_gemm_profile(inventory,wanted['model'],profile,wanted['ln_shared_bytes'])
            graph_non_gemm_checked=True
        if 'expected_graph_parameter_layout' in before:
            from tools.graph_parameter_layout import validate_non_gemm_parameter_layout
            layout=read_output('graph_parameter_layout.json',32*1024*1024)
            if freeze(layout.get('device'))!=freeze(before['expected'].get('device')):
                raise ValueError('parameter layout device mismatch')
            if layout.get('execution_sha256')!=content_hashes['execution.json']:
                raise ValueError('parameter layout execution content mismatch')
            if layout.get('inventory_sha256')!=content_hashes['graph_kernel_inventory.json']:
                raise ValueError('parameter layout inventory content mismatch')
            if 'expected_graph_gemm_parameter_storage' in before:
                validate_non_gemm_parameter_layout(layout,inventory,before['expected'],
                    before['expected_graph_parameter_layout'],before.get('expected_graph_attention_parameter_storage'),
                    before['expected_graph_gemm_parameter_storage'])
                graph_gemm_parameter_layout_checked=True
                graph_attention_parameter_layout_checked='expected_graph_attention_parameter_storage' in before
            elif 'expected_graph_attention_parameter_storage' in before:
                validate_non_gemm_parameter_layout(layout,inventory,before['expected'],
                    before['expected_graph_parameter_layout'],before['expected_graph_attention_parameter_storage'])
                graph_attention_parameter_layout_checked=True
            else:
                validate_non_gemm_parameter_layout(layout,inventory,before['expected'],before['expected_graph_parameter_layout'])
            graph_parameter_layout_checked=True
        if 'compiled_probe_bundle' in before:
            if not all((graph_parameter_layout_checked,graph_scalars_checked,graph_pointers_checked,
                graph_dims_checked,graph_network_checked,graph_attention_members_checked,
                graph_gemm_public_iterators_checked,graph_gemm_mainloop_iterators_checked,
                graph_gemm_launch_values_checked)):
                raise ValueError('missing independent checks required for composed argument coverage')
            from tools.graph_argument_coverage import validate_argument_coverage
            argument_coverage=validate_argument_coverage(inventory,pointer_values,scalar_values,dims_values,
                network,covered_gemm_members,covered_attention_members,
                empty_arguments_checked='expected_empty_elementwise_arguments' in before)
        # Binding is not independent symbol/profile validation or replay proof.
        # Keep this distinct from kernel coverage and production admission.
        graph_inventory_checked=True
    resolved_checked=False
    if 'expected_resolved' in before:
        from tools.resolved_execution_contract import validate_resolved_receipt
        resolved=read_output('resolved_execution.json',8*1024*1024)
        validate_resolved_receipt(resolved,before['expected_resolved'],before['expected'],
                                  execution,content_hashes['raw_scores.json'])
        resolved_checked=True
    replay=None
    if 'expected_cupti_replay' in before:
        from tools.cupti_replay_trace import validate_replay_trace,observe_injection_library
        if observe_injection_library(before['expected_cupti_replay']['path'])!=before['expected_cupti_replay']:
            raise ValueError('CUPTI injection library/parser changed during scorer execution')
        trace_path=run_dir/'runner.log'
        if trace_path.stat().st_size>32*1024*1024:
            raise ValueError('CUPTI activity trace exceeds bounded size')
        replay=validate_replay_trace(trace_path.read_text(),inventory,transfers,
            device_ordinal=before['expected']['device']['device'])
        replay['trace_sha256']=hashlib.sha256(trace_path.read_bytes()).hexdigest()
        replay['injection']=before['expected_cupti_replay']
    if freeze(observe_inputs())!=before_json:
        raise ValueError('candidate inputs/device/profile changed during scorer execution')
    result=inspect_candidate(reference,scores,execution,before['expected'])
    result.update(inputs_unchanged=True)
    # Bind a later scoped admission decision to the actual frozen inputs and
    # every parsed output, rather than trusting a free-standing pass boolean.
    result['observed_bindings']=before
    result['candidate_content_hashes']=content_hashes
    if replay is not None:
        result['executed_graph_replay_checked']=True
        result['executed_graph_replay']=replay
    if graph_inventory_checked:
        result['graph_inventory_bindings_checked']=True
        result['graph_inventory_structure_checked']=True
    if graph_gemm_public_iterators_checked:
        result['graph_gemm_public_iterators_checked']=True
    if graph_gemm_launch_values_checked:
        result['graph_gemm_launch_values_checked']=True
        result['graph_gemm_tensor_iterators_checked']=True
    if graph_gemm_mainloop_iterators_checked:
        result['graph_gemm_mainloop_iterators_checked']=True
    if graph_gemm_checked:
        result['graph_gemm_profile_checked']=True
    if graph_attention_checked:
        result['graph_attention_profile_checked']=True
    if graph_non_gemm_checked:
        result['graph_non_gemm_profile_checked']=True
    if graph_parameter_layout_checked:
        result['graph_non_gemm_parameter_layout_checked']=True
    if graph_attention_parameter_layout_checked:
        result['graph_attention_parameter_layout_checked']=True
    if graph_gemm_parameter_layout_checked:
        result['graph_gemm_parameter_layout_checked']=True
    if graph_transfers_checked:
        result['graph_transfer_arguments_checked']=True
    if graph_transfer_pitched_geometry_checked:
        result['graph_transfer_pitched_geometry_checked']=True
    if graph_dependencies_checked:
        result['graph_dependency_order_checked']=True
    if graph_scalars_checked:
        result['graph_non_gemm_u32_values_checked']=True
    if graph_pointers_checked:
        result['graph_known_pointer_roles_checked']=True
    if graph_dims_checked:
        result['graph_embedded_dims_checked']=True
    if graph_network_checked:
        result['graph_consumed_network_members_checked']=True
    if graph_attention_members_checked:
        result['graph_attention_members_checked']=True
    if resolved_checked:
        result['resolved_bindings_checked']=True
    if 'expected_empty_elementwise_arguments' in before:
        result['canonical_empty_elementwise_arguments_checked']=True
    if 'compiled_probe_bundle' in before:
        result['public_parameter_field_storage_checked']=True
        result['nested_gemm_storage_checked']=True
        result['graph_argument_coverage']=argument_coverage
    if _replay_saved:return result
    temporary=run_dir/'comparison.json.tmp'
    with temporary.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary,run_dir/'comparison.json')
    return result


def validate_saved_candidate(run_dir, bindings, reference):
    """Recheck every saved raw contract and trace without launching or writing.

    The caller must separately bind current source, binary, library, model,
    reference and hardware. This does not create a fresh execution receipt.
    """
    return collect_fresh_candidate([],run_dir,1,lambda:bindings,reference,_replay_saved=True)


def run_scorer(command, log_path, timeout, *, cwd=None, env=None):
    if os.name != 'posix' or not math.isfinite(timeout) or not 0 < timeout <= 3600:
        raise ValueError('scorer requires POSIX and bounded positive timeout')
    process, pending, spawning = None, None, False

    def interrupt(number, frame):
        nonlocal pending
        if spawning:
            pending = number
        else:
            raise KeyboardInterrupt('scorer supervision interrupted')

    previous = {number: signal.signal(number, interrupt)
                for number in (signal.SIGINT, signal.SIGTERM)}
    try:
        with log_path.open('x') as log:
            spawning = True
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, cwd=cwd, env=env)
            spawning = False
            if pending is not None:
                raise KeyboardInterrupt('scorer supervision interrupted during spawn')
            try:
                return process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('production scorer timed out') from error
    finally:
        # Own only this fresh process group, including descendants whose parent
        # exited; never stop another workload or the cloud instance here.
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        try:
            if process is not None:
                stop_group(process)
        finally:
            for number, handler in previous.items():
                signal.signal(number, handler)


def inspect_candidate(reference, scores, execution, expected):
    def same_typed(actual, wanted):
        if type(actual) is not type(wanted):
            return False
        if isinstance(wanted, dict):
            return actual.keys() == wanted.keys() and all(
                same_typed(actual[key], value) for key, value in wanted.items())
        if isinstance(wanted, list):
            return len(actual) == len(wanted) and all(
                same_typed(left, right) for left, right in zip(actual, wanted))
        return actual == wanted

    fields = set(expected) | {'schema_version', 'scope', 'production_quality_accepted',
        'resolved_execution_contract_complete', 'loaded_tensor_content_attestation_complete',
        'device_upload_bytes_verified', 'communicator_initialized', 'search_buffers_allocated',
        'lanes_exercised_serially', 'distinct_reference_rows', 'loaded_view',
        'host_kernel_observations', 'kernel_coverage_complete', 'cohort_generalization_proven',
        'scores_before_quantization'}
    if (not isinstance(execution, dict) or set(execution) != fields or
            type(execution['schema_version']) is not int or execution['schema_version'] != 1 or
            execution['scope'] != 'actual_runner_raw_score_candidate_not_admission'):
        raise ValueError('production candidate schema mismatch')
    for field, value in expected.items():
        if not same_typed(execution[field], value):
            raise ValueError('production candidate binding mismatch: ' + field)
    for field in ('production_quality_accepted', 'resolved_execution_contract_complete',
                  'kernel_coverage_complete', 'cohort_generalization_proven',
                  'communicator_initialized', 'search_buffers_allocated'):
        if execution[field] is not False:
            raise ValueError('unvalidated candidate promotion or scope: ' + field)
    for field in ('loaded_tensor_content_attestation_complete', 'device_upload_bytes_verified',
                  'scores_before_quantization', 'lanes_exercised_serially'):
        if execution[field] is not True:
            raise ValueError('missing actual candidate execution evidence: ' + field)
    if (type(execution['distinct_reference_rows']) is not int or
            execution['distinct_reference_rows'] != len(reference)):
        raise ValueError('candidate reference cohort count mismatch')
    result = compare_production_raw_jobs(reference, scores,
        outer_microbatch=expected['outer_microbatch'], lanes=expected['lanes'],
        atol=.05, rtol=.001, top_k=4, min_topk_overlap=.75)
    if result['status'] != 'pass':
        raise ValueError('production raw-score numeric/ranking comparison failed')
    return dict(status='numeric_pass_unadmitted', production_admitted=False,
        comparison=result, missing_admission=['complete resolved execution contract',
            'complete kernel coverage validator', 'complete independently validated production admission'])


def main():
    """Fresh per-device evidence; incomplete candidates cannot start beam ranks."""
    import argparse
    from pathlib import Path
    import re
    import sys
    from tools.cuda_device_observation import observe
    from tools.cube4_numeric_gate import read_json,validate_reference,validate_checkpoint_binding
    from tools.cube4_bundle_contract import validate_cube4_weights
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('runner','probe','reference-dir','output-root','source-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--world-size',type=int,required=True)
    parser.add_argument('--timeout',type=float,default=120)
    parser.add_argument('--cupti-library',type=Path)
    args=parser.parse_args()
    previous={key:os.environ.get(key) for key in ('LOCAL_RANK','RANK','WORLD_SIZE')}
    try:
        if not 1<=args.world_size<=8 or not math.isfinite(args.timeout) or not 0<args.timeout<=3600:
            raise ValueError('invalid numerical preflight bounds')
        runner=args.runner.resolve(strict=True)
        probe=args.probe.resolve(strict=True)
        source=args.source_root.resolve(strict=True)
        reference_dir=args.reference_dir.resolve(strict=True)
        weight_dir=Path(os.environ.get('BEAM_WEIGHT_DIR',str(reference_dir/'weights_fp16'))).resolve(strict=True)
        def u32(key,default):
            text=os.environ.get(key) or str(default)
            if not re.fullmatch('[0-9]{1,10}',text): raise ValueError('invalid batching: '+key)
            value=int(text)
            if not 0<value<=65536: raise ValueError('batching bound: '+key)
            return value
        outer=u32('BEAM_B_MICRO',8192)
        inner=u32('BEAM_STREAM1_TRANSFORMER_MICRO',outer)
        lanes=u32('BEAM_STREAM1_CONCURRENCY',1)
        executor=os.environ.get('BEAM_STREAM1_EXECUTOR') or 'native_cuda_graph'
        if args.cupti_library is not None and executor!='native_cuda_graph':
            raise ValueError('CUPTI replay proof requires native_cuda_graph')
        if executor not in ('native_cuda_graph','native_eager') or inner>outer or lanes>64:
            raise ValueError('unsupported numerical preflight executor/profile')
        selector=source/'cuda/stream1_transformer_policy_snapshot.hpp'
        keys=re.findall(r'"(BEAM_[A-Z0-9_]+)"',selector.read_text())
        if len(keys)!=26 or len(set(keys))!=26: raise ValueError('selector inventory drift')
        reference,_=read_json(reference_dir/'reference.json')
        expected_scores=validate_reference(reference)
        reports=[]
        # Exclusively owned fresh directory. Never accept a previous receipt.
        args.output_root.mkdir(exist_ok=False)
        compiled_probes=None
        if executor=='native_cuda_graph':
            from tools.storage_probe_bundle import build_storage_probes,observe_storage_probes
            compiled_probes=build_storage_probes(source,runner.parent,args.output_root,args.timeout)
            from tools.transfer_geometry_observation import build_geometry_probe
            build_geometry_probe(source,runner.parent,args.output_root,args.timeout)
            from tools.elementwise_arguments_observation import build_empty_arguments_probe
            build_empty_arguments_probe(source,runner.parent,args.output_root,args.timeout)
        for rank in range(args.world_size):
            os.environ.update(LOCAL_RANK=str(rank),RANK=str(rank),WORLD_SIZE='1')
            def observe_inputs():
                observed_reference,reference_hash=read_json(reference_dir/'reference.json')
                scores=validate_reference(observed_reference)
                manifest,manifest_hash=read_json(weight_dir/'manifest.json')
                validate_checkpoint_binding(observed_reference,manifest)
                prepared=validate_cube4_weights(weight_dir)
                mapping=observe(probe)
                devices=mapping['visible_devices']['devices']
                if len(devices)<args.world_size: raise ValueError('insufficient CUDA-visible devices')
                device=dict(schema_version=1,production_quality_accepted=False,**devices[rank])
                expected=dict(reference_sha256=reference_hash,manifest_sha256=manifest_hash,
                    outer_microbatch=outer,transformer_microbatch=inner,lanes=lanes,
                    executor=executor,device=device,loaded_tensor_bindings=manifest['tensor_files'],
                    requested_launch_policies={key:os.environ.get(key) for key in keys})
                from tools.resolved_execution_contract import canonical_resolved_recipe,observe_recipe_sources
                expected_resolved=canonical_resolved_recipe(expected)
                observed=dict(expected_resolved=expected_resolved,
                    resolved_recipe_sources=observe_recipe_sources(source),
                    reference_dir=str(reference_dir),reference_scores=scores,expected=expected,
                    require_graph_inventory=executor=='native_cuda_graph',
                    require_graph_transfers=executor=='native_cuda_graph',
                    require_graph_dependencies=executor=='native_cuda_graph',prepared=prepared,
                    devices=mapping,runner_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(),
                    selector_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),
                    effective_environment={k:v for k,v in os.environ.items() if k.startswith(('BEAM_','CUDA_'))})
                if args.cupti_library is not None:
                    from tools.cupti_replay_trace import observe_injection_library
                    observed['expected_cupti_replay']=observe_injection_library(args.cupti_library)
                if executor=='native_cuda_graph':
                    from tools.network_member_consumption import observe_network_consumption_sources
                    observed['network_consumption_sources']=observe_network_consumption_sources(source)
                    from tools.elementwise_arguments_observation import observe_empty_arguments_probe
                    observed['expected_empty_elementwise_arguments']=observe_empty_arguments_probe(source,runner.parent)
                    from tools.transfer_geometry_observation import observe_geometry_probe
                    observed['expected_graph_transfer_geometry']=observe_geometry_probe(source,runner.parent,expected)
                    compact=os.environ.get('BEAM_STREAM1_TRANSFORMER_COMPACT57') or '0'
                    if compact not in ('0','1'): raise ValueError('invalid compact sequence selector')
                    observed['expected_graph_gemm_profile']=dict(model=manifest,
                        padded_seq_len=57 if compact=='1' else 64)
                    current_probes=observe_storage_probes(source,runner.parent)
                    if current_probes!=compiled_probes:raise ValueError('compiled probe bundle changed before/during scorer')
                    observed['compiled_probe_bundle']=current_probes
                    tables=current_probes['tables'];padded=57 if compact=='1' else 64
                    observed['expected_graph_gemm_iterator_geometry']=tables['gemm_iterator']
                    observed['expected_graph_gemm_mainloop_geometry']=tables['gemm_iterator']
                    observed['expected_graph_gemm_launch_profile']=dict(
                        geometry=tables['gemm_iterator'],padded_seq_len=padded)
                    observed['expected_graph_attention_profile']=dict(model=manifest,padded_seq_len=padded,
                        storage=tables['attention_geometry'])
                    observed['expected_graph_non_gemm_profile']=dict(model=manifest,padded_seq_len=padded,
                        ln_shared_bytes=tables['parameter']['ln_shared_bytes'])
                    observed['expected_graph_non_gemm_scalar_profile']=dict(model=manifest,padded_seq_len=padded,
                        ln_shared_bytes=tables['parameter']['ln_shared_bytes'])
                    observed['expected_graph_pointer_profile']=dict(model=manifest,padded_seq_len=padded,
                        ln_shared_bytes=tables['parameter']['ln_shared_bytes'])
                    observed['expected_graph_dims_profile']=dict(model=manifest,padded_seq_len=padded,
                        ln_shared_bytes=tables['parameter']['ln_shared_bytes'])
                    observed['expected_graph_network_profile']=dict(model=manifest,padded_seq_len=padded,
                        ln_shared_bytes=tables['parameter']['ln_shared_bytes'])
                    observed['expected_graph_attention_members_profile']=dict(model=manifest,padded_seq_len=padded)
                    observed['expected_graph_parameter_layout']=tables['parameter']
                    observed['expected_graph_attention_parameter_storage']=tables['attention_parameter']
                    if device['sm']==75:
                        observed['expected_graph_gemm_parameter_storage']=tables['gemm_parameter']
                    elif device['sm'] in (80,86):
                        observed['expected_graph_gemm_parameter_storage']=tables['gemm_parameter_sm80']
                return observed
            report=collect_fresh_candidate([str(runner)],args.output_root/f'device{rank}',
                args.timeout,observe_inputs,expected_scores)
            reports.append(report)
        result=dict(schema_version=1,scope='fresh_per_device_candidates_not_admission',
            production_admitted=False,status='numeric_pass_unadmitted',devices=reports)
        with (args.output_root/'summary.json').open('x') as stream:
            json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(result,indent=2,allow_nan=False))
        # No exit0 until complete independent admission is implemented.
        print('numerical candidates verified; complete production admission unavailable',file=sys.stderr)
        return 2
    except (ValueError,OSError,KeyError,TypeError,RuntimeError,subprocess.SubprocessError) as error:
        print('production preflight rejected: '+str(error),file=sys.stderr)
        return 2
    finally:
        for key,value in previous.items():
            if value is None: os.environ.pop(key,None)
            else: os.environ[key]=value


if __name__=='__main__':
    raise SystemExit(main())
