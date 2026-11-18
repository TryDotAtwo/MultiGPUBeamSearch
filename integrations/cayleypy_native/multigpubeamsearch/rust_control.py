"""Coarse adapters to the Rust owner; model export remains in PyTorch.

The comparative migration tests call these adapters directly. Public launch
switching is a separate acceptance stage.
"""
from dataclasses import asdict
import importlib
import json
import os
from pathlib import Path
from .errors import NativeBackendError, NativeUnavailable


def extension():
    import_error = None
    for name in ('multigpubeamsearch._beam_runtime', '_beam_runtime'):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                import_error = error
            continue
        except ImportError as error:
            import_error = error
            continue
        if getattr(module, 'ABI_VERSION', None) != 1 or not hasattr(module, 'InferenceCalibration'):
            raise NativeUnavailable('Rust runtime is incompatible; rebuild the matching PyO3 module')
        return module
    if import_error is not None:
        raise NativeUnavailable('the matching Rust/PyO3 runtime could not be loaded') from import_error
    raise NativeUnavailable('the matching Rust/PyO3 runtime is not installed')


def worker_executable():
    """Return this distribution's worker without loading the CUDA engine."""
    name = 'beam-runner.exe' if os.name == 'nt' else 'beam-runner'
    path = Path(__file__).resolve().parent / '_bin' / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise NativeUnavailable('the matching Rust worker is missing or not executable; reinstall the package')
    return path


def options_descriptor(options):
    values = asdict(options)
    # Omission is also the Rust default, and keeps OFF requests readable by a
    # previously installed Rust extension that predates the experimental key.
    if values['experimental_index_sort'] is False:
        values.pop('experimental_index_sort')
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in values.items()}


def _native_options_supported(options):
    # Keep Python's historical acceptance of values that its public validator
    # permits but beam-core cannot represent in its fixed-width fields.
    if type(options.warn_on_fallback) is not bool:
        return False
    if type(options.num_gpus) is int and options.num_gpus > 65535:
        return False
    if options.devices is not None and (
        len(options.devices) > 65535
        or any(type(device) is int and device > (1 << 32) - 1 for device in options.devices)
    ):
        return False
    if (
        type(options.calibration_frontier_max_states) is int
        and options.calibration_frontier_max_states >= 1 << 64
    ):
        return False
    for name in (
        'timeout_seconds',
        'build_timeout_seconds',
        'calibration_seconds',
        'calibration_pipeline_seconds',
    ):
        value = getattr(options, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= float(1 << 64):
            return False
    return True


def validate_options(options):
    """Validate public options in beam-core when the matching API is present."""
    if not _native_options_supported(options):
        return False
    try:
        payload = json.dumps(options_descriptor(options), allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return False
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return False
    validator = getattr(native, 'validate_options', None)
    if validator is None:
        return False
    try:
        validator(payload)
    except ValueError:
        # Rust decides validity for supported configurations. For rejected
        # inputs, run the compatibility validator so existing Python error
        # messages remain stable at the public API boundary.
        options._validate_python()
        raise
    return True


def _tag_search_value(value):
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is float:
        return {"kind": "float", "value": repr(value)}
    if value is None:
        return {"kind": "none", "value": None}
    return {"kind": "other", "value": None}


def validate_search_controls(params):
    """Delegate normalized public search-control checks to beam-core when available."""
    beam_mode = params["beam_mode"]
    history_depth = params["history_depth"]
    # Python permits user-defined equality for these two legacy fields. Keep
    # that edge case on the compatibility path instead of changing its result.
    if type(beam_mode) not in (str, bool, int, float) and beam_mode is not None:
        return False
    if type(history_depth) not in (str, bool, int, float) and history_depth is not None:
        return False
    request = {
        "beam_width": _tag_search_value(params["beam_width"]),
        "max_steps": _tag_search_value(params["max_steps"]),
        "beam_mode": _tag_search_value(beam_mode),
        "history_depth": _tag_search_value(history_depth),
        "destination_state_present": params["destination_state"] is not None,
        "bfs_result_for_mitm_present": params["bfs_result_for_mitm"] is not None,
        "return_path": _tag_search_value(params["return_path"]),
        "use_child_scores": _tag_search_value(params["use_child_scores"]),
    }
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return False
    validator = getattr(native, "validate_search_controls", None)
    if validator is None:
        return False
    try:
        status, message = validator(json.dumps(request, allow_nan=False))
    except (TypeError, ValueError, OverflowError):
        return False
    if status == "valid":
        return True
    if status == "invalid":
        raise ValueError(message)
    if status == "unsupported":
        raise NativeUnavailable(message)
    raise NativeBackendError("Rust search-control validator returned an unknown status")


def build_graph_contract(center, start, generators, generator_names):
    """Build a validated permutation contract when the extension supports it."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "build_graph_contract", None)
    if builder is None:
        return None
    request = {
        "center": [str(value) for value in center],
        "start": [str(value) for value in start],
        "generators": [[str(value) for value in row] for row in generators],
        "generator_name_count": len(generator_names),
    }
    try:
        return json.loads(builder(json.dumps(request, allow_nan=False)))
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error



def _inference_request(contract, model, runtime, options, devices, beam_width,
                       run_dir, environment, *, capacity_bootstrap=False):
    from .autotune import calibration_signature
    from .build import file_sha256
    from .models import verify_prepared_model
    from .shared_profiles import lookup
    verify_prepared_model(model, contract)
    helper = runtime.runner.parent / runtime.build_metadata.get(
        'calibration_binary_name', 'stream1_ensemble_benchmark')
    files = []
    for name in model.artifact_files:
        path = model.weights_dir / name
        files.append(dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size))
    if model.backend == 'ensemble':
        manifest = dict(model.manifest['ensemble'])
        manifest['models'] = [dict(entry, weights_dir=str(model.weights_dir / entry['weights_dir']))
                              for entry in manifest['models']]
    else:
        manifest = dict(weights_dir=str(model.weights_dir))
    request = dict(options=options_descriptor(options),
        graph=dict(center=contract.center, generators=contract.generators),
        helper=dict(path=str(helper), sha256=runtime.build_metadata.get('calibration_binary_sha256'), bytes=None),
        artifacts=files, probe_manifest=manifest, ensemble=model.backend == 'ensemble',
        signature=calibration_signature(contract, model, runtime, devices, beam_width),
        beam_width=beam_width, protocol=runtime.build_metadata.get('calibration_protocol', 'oneshot'),
        capacity_bootstrap=capacity_bootstrap, run_dir=str(Path(run_dir).resolve()),
        environment=dict(environment), shared_profile=None if capacity_bootstrap else
            lookup(contract, model, runtime, options, devices, beam_width))
    return request


def tune_inference(contract, model, runtime, options, devices, beam_width,
                   run_dir, environment, *, capacity_bootstrap=False):
    from .models import verify_prepared_model
    request = _inference_request(contract, model, runtime, options, devices, beam_width,
                                 run_dir, environment, capacity_bootstrap=capacity_bootstrap)
    try:
        result = json.loads(extension().InferenceCalibration(json.dumps(request)).run())
    except (RuntimeError, ValueError) as error:
        raise NativeBackendError(str(error)) from error
    verify_prepared_model(model, contract)
    return result


def calibrate_frontier(contract, model, runtime, options, devices, beam_width,
                       run_dir, environment, runner):
    """Transfer one job; Rust owns all admission, calibration and width loops."""
    import torch
    from .models import verify_prepared_model
    initial = 1_048_576 if beam_width == 'max' else beam_width
    inference = _inference_request(contract, model, runtime, options, devices, initial,
        run_dir, environment, capacity_bootstrap=beam_width == 'max')
    hardware = [torch.cuda.get_device_properties(device) for device in devices]
    execution = dict(runner=dict(path=str(Path(runner).resolve()),
            sha256=runtime.build_metadata['binary_sha256'], bytes=None),
        row_factor=contract.move_count if model.backend == 'mlp' and model.manifest['output_dim'] == 1 else 1,
        hardware=dict(sm_count=min(p.multi_processor_count for p in hardware),
            l2_bytes=min(getattr(p, 'L2_cache_size', getattr(p, 'l2_cache_size', 0)) for p in hardware)),
        physical_bytes=runtime.build_metadata['shape']['storage_len'], puzzle_id=0,
        grow=runtime.build_metadata.get('frontier_calibration_protocol') == 'natural-growth-v1',
        fixtures=[], cwd=str(Path.cwd()), gpu_memory_bytes=[p.total_memory for p in hardware])
    try:
        result = json.loads(extension().FrontierCalibration(json.dumps(dict(
            beam_width=beam_width, inference=inference, execution=execution))).run())
    except (RuntimeError, ValueError) as error:
        raise NativeBackendError(str(error)) from error
    verify_prepared_model(model, contract)
    return result


def tune_downstream(contract, model, runtime, options, devices, beam_width,
                    run_dir, environment, runner, inference):
    import torch
    from .build import file_sha256
    from .models import verify_prepared_model
    verify_prepared_model(model, contract)
    files = []
    for name in model.artifact_files:
        path = model.weights_dir / name
        files.append(dict(path=str(path), sha256=file_sha256(path), bytes=path.stat().st_size))
    hardware = [torch.cuda.get_device_properties(device) for device in devices]
    row_factor = contract.move_count if model.backend == 'mlp' and model.manifest['output_dim'] == 1 else 1
    request = dict(options=options_descriptor(options),
        graph=dict(center=contract.center, generators=contract.generators),
        runner=dict(path=str(Path(runner).resolve()), sha256=runtime.build_metadata['binary_sha256'], bytes=None),
        artifacts=files, inference=inference, environment=dict(environment), beam_width=beam_width,
        row_factor=row_factor, hardware=dict(sm_count=min(p.multi_processor_count for p in hardware),
            l2_bytes=min(getattr(p, 'L2_cache_size', getattr(p, 'l2_cache_size', 0)) for p in hardware)),
        run_dir=str(Path(run_dir).resolve()), cwd=str(Path.cwd()),
        physical_bytes=runtime.build_metadata['shape']['storage_len'], puzzle_id=0,
        grow=runtime.build_metadata.get('frontier_calibration_protocol') == 'natural-growth-v1', fixtures=[])
    try:
        result = json.loads(extension().DownstreamCalibration(json.dumps(request)).run())
    except (RuntimeError, ValueError) as error:
        raise NativeBackendError(str(error)) from error
    verify_prepared_model(model, contract)
    return result


