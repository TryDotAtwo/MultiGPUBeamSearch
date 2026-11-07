"""Coarse adapters to the Rust owner; model export remains in PyTorch.

The comparative migration tests call these adapters directly. Public launch
switching is a separate acceptance stage.
"""
from dataclasses import asdict
import importlib
import json
from pathlib import Path
from .errors import NativeBackendError, NativeUnavailable


def extension():
    for name in ('multigpubeamsearch._beam_runtime', '_beam_runtime'):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as error:
            if error.name != name:
                raise
            continue
        if getattr(module, 'ABI_VERSION', None) != 1 or not hasattr(module, 'InferenceCalibration'):
            raise NativeUnavailable('Rust runtime is incompatible; rebuild the matching PyO3 module')
        return module
    raise NativeUnavailable('the matching Rust/PyO3 runtime is not installed')


def options_descriptor(options):
    return {name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(options).items()}


def tune_inference(contract, model, runtime, options, devices, beam_width,
                   run_dir, environment, *, capacity_bootstrap=False):
    from .autotune import calibration_signature
    from .build import file_sha256
    from .models import verify_prepared_model
    from .shared_profiles import lookup
    owner = extension()
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
    try:
        result = json.loads(owner.InferenceCalibration(json.dumps(request)).run())
    except (RuntimeError, ValueError) as error:
        raise NativeBackendError(str(error)) from error
    verify_prepared_model(model, contract)
    return result
