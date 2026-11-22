"""Coarse adapters to the Rust owner; model export remains in PyTorch.

The comparative migration tests call these adapters directly. Public launch
switching is a separate acceptance stage.
"""
from dataclasses import asdict
import importlib
import json
import math
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


def _supports_preparation_identity_schema(native):
    version = getattr(native, 'PREPARATION_IDENTITY_SCHEMA_VERSION', 0)
    return type(version) is int and version >= 2


def supports_preparation_identity():
    """Whether the extension uses structured preparation options (schema v2+)."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return False
    return _supports_preparation_identity_schema(native)


def supports_auto_max_preparation_identity():
    """Whether this extension can decode auto-max in preparation schema v2+."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return False
    if not _supports_preparation_identity_schema(native):
        return False
    version = getattr(native, 'PREPARATION_IDENTITY_AUTO_MAX_VERSION', 0)
    return type(version) is int and version >= 1


def build_preparation_identity(*, graph_hash, geometry, center, generators,
                               model_artifact_hash, artifact_sizes, model_backend,
                               devices, binding_scope, beam_width, max_steps,
                               options, cuda_visible_devices, cuda_device_order,
                               runtime_build, resolved_inference_plan=None):
    """Build the typed resident-session identity in Rust when supported."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    if not _supports_preparation_identity_schema(native):
        return None
    builder = getattr(native, 'build_preparation_identity', None)
    if not callable(builder) or type(options) is not dict:
        return None
    try:
        options_payload = dict(options)
        if resolved_inference_plan is not None:
            if (
                type(resolved_inference_plan) is not dict
                or set(resolved_inference_plan) != {
                    "backend", "policy", "inference_row_limit"
                }
                or resolved_inference_plan.get("backend") not in ("libtorch", "cutlass")
                or type(resolved_inference_plan.get("policy")) is not str
                or not resolved_inference_plan["policy"]
                or type(resolved_inference_plan.get("inference_row_limit")) is not int
                or resolved_inference_plan["inference_row_limit"] <= 0
            ):
                raise ValueError("invalid resolved inference plan")
            options_payload["resolved_inference_plan"] = dict(resolved_inference_plan)
        options_json = json.dumps(options_payload, sort_keys=True, allow_nan=False)
        runtime_build_json = (None if runtime_build is None
                              else json.dumps(runtime_build, sort_keys=True, allow_nan=False))
        if type(beam_width) is str and beam_width == 'max':
            version = getattr(native, 'PREPARATION_IDENTITY_AUTO_MAX_VERSION', 0)
            if type(version) is not int or version < 1:
                return None
            native_beam_width = beam_width
        else:
            native_beam_width = int(beam_width)
        result = builder(
            graph_hash,
            [None if value is None else int(value) for value in geometry],
            None if center is None else [int(value) for value in center],
            None if generators is None else [
                [int(value) for value in move] for move in generators],
            model_artifact_hash,
            None if artifact_sizes is None else [int(size) for size in artifact_sizes],
            model_backend,
            [int(device) for device in devices],
            bool(binding_scope),
            native_beam_width,
            int(max_steps),
            options_json,
            cuda_visible_devices,
            cuda_device_order,
            runtime_build_json,
        )
    except (TypeError, ValueError, OverflowError):
        # Older Python-compatible values remain on the compatibility path.
        return None
    return result if isinstance(result, str) else None
def build_session_identity(*, center, generators, graph_hash, model_artifact_hash,
                           model_backend, devices, beam_width, max_steps,
                           runtime_build, options, cuda_visible_devices,
                           cuda_device_order):
    """Build the typed resident launch identity in Rust when supported."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, 'build_session_identity', None)
    if not callable(builder):
        return None
    try:
        result = builder(
            [int(value) for value in center],
            [[int(value) for value in move] for move in generators],
            graph_hash,
            model_artifact_hash,
            model_backend,
            [int(device) for device in devices],
            int(beam_width),
            int(max_steps),
            json.dumps(runtime_build),
            json.dumps(options),
            cuda_visible_devices,
            cuda_device_order,
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return result if isinstance(result, str) else None

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


def resolve_runtime_devices(visible_device_count, explicit_devices, num_gpus, graph_devices):
    """Resolve process-local device indices in beam-runtime when supported."""
    if type(visible_device_count) is not int or visible_device_count < 0:
        raise NativeUnavailable("visible CUDA device count must be a nonnegative integer")
    if explicit_devices is not None:
        if type(explicit_devices) not in (tuple, list) or any(
            type(device) is not int for device in explicit_devices
        ):
            raise NativeUnavailable("explicit CUDA devices must be an integer sequence")
        explicit = list(explicit_devices)
    else:
        explicit = None
    if num_gpus is not None and type(num_gpus) is not int:
        raise NativeUnavailable("num_gpus must be an integer")
    observations = []
    try:
        for observation in graph_devices:
            if (
                type(observation) not in (tuple, list)
                or len(observation) != 2
                or type(observation[0]) is not str
                or (observation[1] is not None and type(observation[1]) is not int)
            ):
                raise ValueError("invalid normalized graph device observation")
            observations.append((observation[0], observation[1]))
    except TypeError as error:
        raise NativeUnavailable("graph devices must be a sequence of observations") from error
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_runtime_devices", None)
    if not callable(resolver):
        return None
    try:
        selected = resolver(visible_device_count, explicit, num_gpus, observations)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if (
        type(selected) is not list
        or not selected
        or len(selected) > 128
        or any(type(device) is not int or not 0 <= device < visible_device_count
               for device in selected)
        or len(set(selected)) != len(selected)
    ):
        raise NativeUnavailable("Rust runtime returned an invalid device selection")
    return tuple(selected)



def resolve_runtime_device_plan(
    cuda_available, visible_device_count, explicit_devices, num_gpus, graph_devices
):
    """Resolve CUDA availability and process-local device selection in beam-runtime."""
    if type(cuda_available) is not bool:
        raise NativeUnavailable("CUDA availability must be a boolean")
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_runtime_device_plan", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_RUNTIME_DEVICE_PLAN_VERSION", None) != 1:
        raise NativeUnavailable("Rust runtime device plan API version is incompatible")
    if not cuda_available:
        try:
            resolver(False, 0, None, None, [])
        except (RuntimeError, ValueError, TypeError, OverflowError) as error:
            raise NativeUnavailable(str(error)) from error
        raise NativeUnavailable("Rust runtime returned a device plan without CUDA")
    if type(visible_device_count) is not int or not 0 <= visible_device_count <= 0xFFFFFFFF:
        raise NativeUnavailable("visible CUDA device count must fit a uint32")
    if explicit_devices is not None:
        if type(explicit_devices) not in (tuple, list) or any(
            type(device) is not int for device in explicit_devices
        ):
            raise NativeUnavailable("explicit CUDA devices must be an integer sequence")
        explicit = list(explicit_devices)
    else:
        explicit = None
    if num_gpus is not None and type(num_gpus) is not int:
        raise NativeUnavailable("num_gpus must be an integer")
    observations = []
    try:
        for observation in graph_devices:
            if (
                type(observation) not in (tuple, list)
                or len(observation) != 2
                or type(observation[0]) is not str
                or (observation[1] is not None and type(observation[1]) is not int)
            ):
                raise ValueError("invalid normalized graph device observation")
            observations.append((observation[0], observation[1]))
    except TypeError as error:
        raise NativeUnavailable("graph devices must be a sequence of observations") from error
    try:
        selected = resolver(
            True, visible_device_count, explicit, num_gpus, observations
        )
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if (
        type(selected) is not list
        or not selected
        or len(selected) > 128
        or any(type(device) is not int or not 0 <= device < visible_device_count
               for device in selected)
        or len(set(selected)) != len(selected)
    ):
        raise NativeUnavailable("Rust runtime returned an invalid device selection")
    return tuple(selected)



def resolve_search_environment(
    puzzle_info, test_csv, weight_dir, run_dir,
    effective_touch_bfs_radius, touch_bfs_max_entries
):
    """Build fixed permutation search variables with the installed Rust runtime."""
    values = (puzzle_info, test_csv, weight_dir, run_dir)
    if any(type(value) is not str for value in values):
        return None
    if (type(effective_touch_bfs_radius) is not int
            or not 0 <= effective_touch_bfs_radius <= 0xFFFFFFFF
            or type(touch_bfs_max_entries) is not int
            or not 0 <= touch_bfs_max_entries <= 0xFFFFFFFFFFFFFFFF):
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_search_environment", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_SEARCH_ENVIRONMENT_VERSION", None) != 1:
        raise NativeUnavailable("Rust search environment API version is incompatible")
    try:
        encoded = resolver(
            puzzle_info, test_csv, weight_dir, run_dir,
            effective_touch_bfs_radius, touch_bfs_max_entries
        )
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if type(encoded) is not str:
        raise NativeUnavailable("Rust runtime returned an invalid search environment")
    try:
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed search environment JSON") from error
    if not isinstance(result, dict) or any(
        type(name) is not str or type(value) is not str
        for name, value in result.items()
    ):
        raise NativeUnavailable("Rust runtime returned an invalid search environment")
    return result



def resolve_search_model_environment(request):
    """Resolve model-family worker variables through the installed Rust runtime."""
    if type(request) is not dict:
        return None
    try:
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_model_search_environment", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_MODEL_SEARCH_ENVIRONMENT_VERSION", None) != 1:
        raise NativeUnavailable("Rust model search environment API version is incompatible")
    try:
        encoded = resolver(payload)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if type(encoded) is not str:
        raise NativeUnavailable("Rust runtime returned an invalid model search environment")
    try:
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed model environment JSON") from error
    allowed = {
        "BEAM_BLEND_DIR",
        "BEAM_WEIGHT_DIR",
        "BEAM_ENSEMBLE_BACKEND",
        "BEAM_STREAM1_EXECUTOR",
        "BEAM_MLP_GRAPH_HASH",
        "BEAM_MLP_ARTIFACT_HASH",
    }
    if not isinstance(result, dict) or any(
        type(name) is not str or type(value) is not str or name not in allowed
        for name, value in result.items()
    ):
        raise NativeUnavailable("Rust runtime returned an invalid model search environment")
    return result


def resolve_search_policy_environment(request):
    """Resolve fixed solution collection variables through beam-runtime."""
    if type(request) is not dict:
        return None
    try:
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_search_policy_environment", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_SEARCH_POLICY_ENVIRONMENT_VERSION", None) != 1:
        raise NativeUnavailable("Rust search policy environment API version is incompatible")
    try:
        encoded = resolver(payload)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if type(encoded) is not str:
        raise NativeUnavailable("Rust runtime returned an invalid search policy environment")
    try:
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed search policy environment JSON") from error
    allowed = {
        "BEAM_SOLVE_BUCKET_MODE",
        "BEAM_SOLVE_BUCKET_EXTRA_DEPTHS",
        "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS",
        "BEAM_SOLVE_BUCKET_RESULT_TSV",
    }
    if not isinstance(result, dict) or any(
        type(name) is not str or type(value) is not str or name not in allowed
        for name, value in result.items()
    ) or set(result) != allowed:
        raise NativeUnavailable("Rust runtime returned an invalid search policy environment")
    return result


def resolve_child_environment(environment, devices, nccl_id_file):
    """Build the filtered native child environment in beam-runtime when supported."""
    if type(environment) is not dict or any(
        type(name) is not str or type(value) is not str
        for name, value in environment.items()
    ):
        return None
    if type(devices) not in (tuple, list) or any(
        type(device) is not int or not 0 <= device <= 0xFFFFFFFF
        for device in devices
    ):
        return None
    if type(nccl_id_file) is not str:
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_child_environment", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_CHILD_ENVIRONMENT_VERSION", None) != 1:
        raise NativeUnavailable("Rust child environment API version is incompatible")
    try:
        encoded = resolver(environment, list(devices), nccl_id_file)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if not isinstance(encoded, str):
        raise NativeUnavailable("Rust runtime returned an invalid child environment")
    try:
        result = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed child-environment JSON") from error
    if not isinstance(result, dict) or any(
        type(name) is not str or type(value) is not str
        for name, value in result.items()
    ):
        raise NativeUnavailable("Rust runtime returned an invalid child environment")
    return result


def resolve_cuda_visibility(devices, inherited):
    """Map selected logical CUDA indices through the inherited visibility list."""
    if type(devices) not in (tuple, list) or any(
        type(device) is not int or not 0 <= device <= 0xFFFFFFFF
        for device in devices
    ):
        return None
    if inherited is not None and type(inherited) is not str:
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_cuda_visibility", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_DEVICE_BINDING_VERSION", None) != 1:
        raise NativeUnavailable("Rust CUDA visibility API version is incompatible")
    try:
        result = resolver(list(devices), inherited)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if type(result) is not str:
        raise NativeUnavailable("Rust runtime returned an invalid CUDA visibility mapping")
    return result


def resolve_cuda_ordinal(local_rank, visible_devices):
    """Resolve a scheduler rank in the process-local CUDA namespace when supported."""
    if type(visible_devices) is not int or visible_devices <= 0:
        raise NativeUnavailable("no visible CUDA devices")
    # Preserve the Python compatibility path for unusual values outside the
    # fixed-width Rust ABI; scheduler_rank itself always produces bounded ints.
    if type(local_rank) is not int or not -(1 << 63) <= local_rank < (1 << 63):
        return None
    if visible_devices >= 1 << 64:
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_cuda_ordinal", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_DEVICE_BINDING_VERSION", None) != 1:
        raise NativeUnavailable("Rust CUDA ordinal API version is incompatible")
    try:
        ordinal = resolver(local_rank, visible_devices)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if type(ordinal) is not int:
        raise NativeUnavailable("Rust runtime returned an invalid CUDA ordinal")
    return ordinal


def resolve_scheduler_rank(environment):
    """Resolve torchrun/Slurm placement in beam-runtime when the wheel supports it."""
    if type(environment) is not dict or any(
        type(name) is not str or type(value) is not str
        for name, value in environment.items()
    ):
        raise NativeUnavailable("scheduler environment must map strings to strings")
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_scheduler_rank", None)
    if not callable(resolver):
        return None
    if getattr(native, "CLUSTER_RANK_VERSION", None) != 1:
        raise NativeUnavailable("Rust scheduler rank API version is incompatible")
    try:
        encoded = resolver(environment)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if not isinstance(encoded, str):
        raise NativeUnavailable("Rust runtime returned an invalid scheduler rank")
    try:
        rank = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed scheduler-rank JSON") from error
    fields = {
        "global_rank", "node_rank", "local_rank", "world_size", "local_world_size",
    }
    if not isinstance(rank, dict) or set(rank) != fields:
        raise NativeUnavailable("Rust runtime returned an invalid scheduler rank")
    if any(type(rank[name]) is not int or not 0 <= rank[name] < (1 << 64) for name in fields):
        raise NativeUnavailable("Rust runtime returned an invalid scheduler rank")
    if (
        not 1 <= rank["world_size"] <= 128
        or not 1 <= rank["local_world_size"] <= rank["world_size"]
        or rank["global_rank"] >= rank["world_size"]
        or rank["node_rank"] >= rank["world_size"]
        or rank["local_rank"] >= rank["local_world_size"]
    ):
        raise NativeUnavailable("Rust runtime returned an invalid scheduler rank")
    return rank


def resolve_cluster_plan(request):
    """Resolve the public multi-node rank layout in beam-runtime when available."""
    fields = {"gpus_per_node", "beam_width", "shard_count", "batch_alignment"}
    if type(request) is not dict or set(request) != fields:
        raise NativeUnavailable("cluster plan request must contain the exact planning fields")
    counts = request["gpus_per_node"]
    if (
        type(counts) not in (tuple, list)
        or not counts
        or any(type(count) is not int or not 0 < count <= (1 << 32) - 1 for count in counts)
    ):
        raise NativeUnavailable("cluster plan gpus_per_node must be positive uint32 values")
    limits = {
        "beam_width": (1, (1 << 64) - 1),
        "shard_count": (1, (1 << 32) - 1),
        "batch_alignment": (1, (1 << 32) - 1),
    }
    for name, (lower, upper) in limits.items():
        value = request[name]
        if type(value) is not int or not lower <= value <= upper:
            raise NativeUnavailable(f"cluster plan {name} must be in [{lower}, {upper}]")
    payload = dict(request, gpus_per_node=list(counts))
    try:
        encoded_request = json.dumps(payload, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable("cluster plan request is not JSON-compatible") from error
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_cluster_plan", None)
    if not callable(resolver):
        return None
    try:
        encoded = resolver(encoded_request)
    except (RuntimeError, ValueError, TypeError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    if not isinstance(encoded, str):
        raise NativeUnavailable("Rust runtime returned an invalid cluster plan")
    try:
        plan = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed cluster-plan JSON") from error
    expected = {
        "ranks", "requested_beam", "effective_beam", "alignment",
        "local_frontier_capacity", "shard_count", "hardware_verified",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise NativeUnavailable("Rust runtime returned an invalid cluster plan")
    ranks = plan["ranks"]
    expected_rank_count = sum(counts)
    rank_fields = {"global_rank", "node_rank", "local_rank"}
    if not isinstance(ranks, list) or len(ranks) != expected_rank_count:
        raise NativeUnavailable("Rust runtime returned an invalid cluster rank layout")
    offset = 0
    for node, count in enumerate(counts):
        for local in range(count):
            rank = ranks[offset]
            if (
                not isinstance(rank, dict)
                or set(rank) != rank_fields
                or any(type(rank[name]) is not int for name in rank_fields)
                or rank != {"global_rank": offset, "node_rank": node, "local_rank": local}
            ):
                raise NativeUnavailable("Rust runtime returned an invalid cluster rank layout")
            offset += 1
    for name in (
        "requested_beam", "effective_beam", "alignment",
        "local_frontier_capacity", "shard_count",
    ):
        if type(plan[name]) is not int or not 0 < plan[name] < (1 << 64):
            raise NativeUnavailable("Rust runtime returned an invalid cluster plan")
    if (
        plan["requested_beam"] != request["beam_width"]
        or plan["effective_beam"] != request["beam_width"]
        or plan["shard_count"] != request["shard_count"]
        or plan["hardware_verified"] is not False
        or plan["local_frontier_capacity"] >= (1 << 32)
    ):
        raise NativeUnavailable("Rust runtime returned an inconsistent cluster plan")
    return plan


def resolve_microbatch_plan(request):
    """Resolve the established inference-row plan in beam-runtime when available."""
    fields = {
        "beam_width", "world_size", "move_count", "output_dim",
        "model_backend", "row_limit",
    }
    if type(request) is not dict or set(request) != fields:
        raise NativeUnavailable("microbatch request must contain the exact typed planning fields")
    integer_limits = {
        "beam_width": (1, (1 << 64) - 1),
        "world_size": (1, (1 << 32) - 1),
        "move_count": (1, (1 << 32) - 1),
        "output_dim": (1, (1 << 32) - 1),
        "row_limit": (1, (1 << 64) - 1),
    }
    for name, (lower, upper) in integer_limits.items():
        value = request[name]
        if type(value) is not int or not lower <= value <= upper:
            raise NativeUnavailable(f"microbatch {name} must be an integer in [{lower}, {upper}]")
    if type(request["model_backend"]) is not str or not request["model_backend"]:
        raise NativeUnavailable("microbatch model_backend must be a nonempty string")
    try:
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable("microbatch request is not JSON-compatible") from error
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_microbatch_plan", None)
    if not callable(resolver):
        return None
    try:
        encoded = resolver(payload)
    except (RuntimeError, ValueError) as error:
        raise NativeUnavailable(str(error)) from error
    if not isinstance(encoded, str):
        raise NativeUnavailable("Rust runtime returned an invalid microbatch plan")
    try:
        plan = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed microbatch-plan JSON") from error
    expected = {
        "native_default_row_budget", "configured_row_budget", "derived_parent_batch",
        "derived_candidates_per_slot", "reference_logical_shard",
        "reference_shard_storage", "adjusted_for_small_beam",
    }
    if not isinstance(plan, dict) or set(plan) != expected:
        raise NativeUnavailable("Rust runtime returned an invalid microbatch plan")
    for name in expected - {"adjusted_for_small_beam"}:
        if type(plan[name]) is not int or not 0 <= plan[name] <= (1 << 64) - 1:
            raise NativeUnavailable("Rust runtime returned an invalid microbatch plan")
    if type(plan["adjusted_for_small_beam"]) is not bool:
        raise NativeUnavailable("Rust runtime returned an invalid microbatch plan")
    if plan["native_default_row_budget"] != request["row_limit"]:
        raise NativeUnavailable("Rust microbatch plan changed the requested row limit")
    return plan


def resolve_inference_backend(request):
    """Resolve the prepared permutation runner in Rust when the wheel supports it.

    Return None only for a missing/older extension so the public adapter can
    preserve the historical Python policy on previously installed wheels.
    """
    if type(request) is not dict:
        raise NativeUnavailable("inference backend request must be an object")
    try:
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable("inference backend request is not JSON-compatible") from error
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_inference_backend", None)
    if not callable(resolver):
        return None
    try:
        selected = resolver(payload)
    except (RuntimeError, ValueError) as error:
        raise NativeUnavailable(str(error)) from error
    if selected not in ("libtorch", "cutlass"):
        raise NativeUnavailable("Rust runtime returned an invalid inference backend")
    return selected


def resolve_inference_plan(request):
    """Resolve the backend and its existing profile in the Rust runtime when available."""
    if type(request) is not dict:
        raise NativeUnavailable("inference backend request must be an object")
    try:
        payload = json.dumps(request, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable("inference backend request is not JSON-compatible") from error
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    resolver = getattr(native, "resolve_inference_plan", None)
    if not callable(resolver):
        return None
    try:
        encoded = resolver(payload)
    except (RuntimeError, ValueError) as error:
        raise NativeUnavailable(str(error)) from error
    if not isinstance(encoded, str):
        raise NativeUnavailable("Rust runtime returned an invalid inference plan")
    try:
        plan = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise NativeUnavailable("Rust runtime returned malformed inference-plan JSON") from error
    if (
        not isinstance(plan, dict)
        or set(plan) != {"backend", "policy", "inference_row_limit"}
        or plan.get("backend") not in ("libtorch", "cutlass")
        or not isinstance(plan.get("policy"), str)
        or not plan["policy"]
        or len(plan["policy"]) > 128
        or type(plan.get("inference_row_limit")) is not int
        or not 1 <= plan["inference_row_limit"] <= 8192
    ):
        raise NativeUnavailable("Rust runtime returned an invalid inference plan")
    return plan


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



def write_search_inputs(contract, run_dir):
    """Write legacy runner task files through Rust when this extension supports it."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return False
    writer = getattr(native, "write_search_inputs", None)
    if not callable(writer):
        return False
    try:
        request = {
            "center": [str(value) for value in contract.center],
            "start": [str(value) for value in contract.start],
            "generators": [[str(value) for value in row] for row in contract.generators],
            "generator_name_count": len(contract.generator_names),
        }
    except AttributeError:
        return False
    try:
        writer(json.dumps(request, allow_nan=False), str(run_dir))
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeUnavailable(str(error)) from error
    return True


def inference_coarse_batches(cap):
    """Use Rust candidate ordering when this extension exposes the optional API."""
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "inference_coarse_batches", None)
    if not callable(builder):
        return None
    try:
        result = builder(cap)
    except (TypeError, ValueError, OverflowError):
        return None
    if type(result) is not list or any(type(value) is not int for value in result):
        return None
    return result


def inference_refinement_batches(winner, cap, visited):
    """Use Rust refinement ordering for ordinary integer candidate sets."""
    if type(visited) not in (set, frozenset, list, tuple):
        return None
    if any(type(value) is not int for value in visited):
        return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "inference_refinement_batches", None)
    if not callable(builder):
        return None
    try:
        result = builder(winner, cap, sorted(set(visited)))
    except (TypeError, ValueError, OverflowError):
        return None
    if type(result) is not list or any(type(value) is not int for value in result):
        return None
    return result


def inference_probe_time_allowance(records, default=12.0):
    """Use the Rust formula for the ordinary timing records emitted by autotune."""
    if type(records) not in (list, tuple) or type(default) is not float:
        return None
    if not math.isfinite(default):
        return None

    normalized = []
    for row in records:
        if type(row) is not dict:
            return None
        try:
            seconds = row.get("seconds")
            if not seconds:
                continue
        except Exception:
            return None
        if type(seconds) not in (list, tuple):
            return None
        if any(type(value) not in (int, float) for value in seconds):
            return None
        try:
            values = [float(value) for value in seconds]
        except (OverflowError, TypeError, ValueError):
            return None
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            return None
        normalized.append(values)

    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "inference_probe_time_allowance", None)
    if not callable(builder):
        return None
    try:
        result = builder(normalized, default)
    except (TypeError, ValueError, OverflowError):
        return None
    if type(result) not in (int, float) or not math.isfinite(result):
        return None
    return float(result)



def inference_capacity_cache_reusable(previous, current):
    """Apply Rust cache policy to ordinary per-GPU memory snapshots."""
    for snapshot in (previous, current):
        if snapshot is None:
            continue
        if type(snapshot) is not dict:
            return None
        if any(
            type(uuid) is not str
            or type(free_memory) is not int
            or free_memory < 0
            or free_memory > 0xFFFF_FFFF_FFFF_FFFF
            for uuid, free_memory in snapshot.items()
        ):
            return None
    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "inference_capacity_cache_reusable", None)
    if not callable(builder):
        return None
    try:
        result = builder(
            json.dumps(previous, sort_keys=True),
            json.dumps(current, sort_keys=True),
        )
    except (TypeError, ValueError, OverflowError):
        return None
    return result if type(result) is bool else None



def inference_select_compatible(samples, baseline):
    """Use Rust only for ordinary Measurement rows; preserve Python fallback."""
    if type(samples) not in (list, tuple) or type(baseline) is not str:
        return None
    from .calibration_stats import Estimate, Measurement

    normalized = []
    for row in samples:
        if type(row) is not Measurement:
            return None
        if (
            type(row.profile) is not str
            or type(row.parents) is not int
            or row.parents <= 0
            or row.parents > (1 << 53)
            or type(row.seconds_by_rank) not in (tuple, list)
            or type(row.correctness_passed) is not bool
            or type(row.memory_admitted) is not bool
            or type(row.throttled) is not bool
        ):
            return None
        if any(type(value) is not float or not math.isfinite(value) or value <= 0.0
               for value in row.seconds_by_rank):
            return None
        if row.seconds_by_rank:
            per_parent = max(row.seconds_by_rank) / row.parents
            if not math.isfinite(per_parent) or per_parent <= 0.0:
                return None
        normalized.append(asdict(row))

    try:
        native = extension()
    except (NativeUnavailable, ImportError):
        return None
    builder = getattr(native, "inference_select_compatible", None)
    if not callable(builder):
        return None
    try:
        request = json.dumps(
            {"baseline": baseline, "samples": normalized},
            allow_nan=False,
            separators=(",", ":"),
        )
        selected, rejected = json.loads(builder(request))
        if type(selected) is not dict or type(rejected) is not dict:
            return None
        if any(type(key) is not str or type(value) is not str
               for key, value in rejected.items()):
            return None
        return Estimate(**selected), rejected
    except (TypeError, ValueError, OverflowError):
        return None


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


