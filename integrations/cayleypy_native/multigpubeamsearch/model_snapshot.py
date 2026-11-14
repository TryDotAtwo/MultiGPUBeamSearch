"""Session-local reuse of a validated private MLP artifact snapshot."""
from copy import deepcopy
from pathlib import Path


def requires_mlp_host_binding(cache, model, contract, build_metadata):
    """Share the exact opt-in predicate between session reuse and launch."""
    return (cache is not None and cache.binds(model, contract) and model.backend == 'mlp'
            and build_metadata.get('inference_backend', 'cutlass') == 'cutlass')


class ModelSnapshotCache:
    """Retain one snapshot; callers serialize access using the session lock."""

    def __init__(self):
        self._entry = None
        self._source = None
        self._prepared = None
        self._active = None
        self.last = None

    def prepare(self, model, contract, run_dir, cache_dir):
        from .models import PreparedModel, _snapshot_artifact, verify_prepared_model
        from .errors import NativeBackendError

        key = (str(Path(cache_dir).resolve()), contract.graph_hash, model.backend, model.artifact_hash)
        self.last = None
        self._source = model
        self._prepared = None
        self._active = None
        if self._entry is not None and self._entry[0] == key:
            # Reconstruct from the freshly validated source rather than a
            # previous public manifest. NativeSession.run checks the retained
            # snapshot once immediately before either submission or reload.
            snapshot = PreparedModel(self._entry[1], deepcopy(model.manifest), model.backend,
                                     model.artifact_hash, model.artifact_files)
            # The source was validated before lookup; retain its second
            # verification so a change during reuse is not hidden.
            verify_prepared_model(model, contract)
            self.last = {'reused': True, 'copied_bytes': 0, 'snapshot_dir': str(snapshot.weights_dir)}
            self._prepared = snapshot
            self._active = snapshot
            return snapshot
        try:
            copied_bytes = sum((model.weights_dir / name).stat().st_size for name in model.artifact_files)
        except OSError as error:
            raise NativeBackendError(f"could not inspect native model snapshot inputs: {error}") from error
        snapshot = _snapshot_artifact(model, contract, run_dir)
        self._entry = (key, snapshot.weights_dir)
        self.last = {'reused': False, 'copied_bytes': copied_bytes, 'snapshot_dir': str(snapshot.weights_dir)}
        self._active = snapshot
        return snapshot

    def binds(self, model, contract):
        """Only the exact snapshot returned for this request opts in to binding."""
        return (self._active is model and self.last is not None and self._entry is not None
                and self._entry[0][1:] == (contract.graph_hash, model.backend, model.artifact_hash))

    def needs_fresh_worker_snapshot(self, model, contract):
        """Bind a refresh to the exact prepared object of this cache hit."""
        return (self._entry is not None and self.last is not None and self.last['reused']
                and self._prepared is model and self._entry[0][1:] ==
                (contract.graph_hash, model.backend, model.artifact_hash))

    def refresh_for_worker(self, model, contract, run_dir):
        """New workers read a fresh snapshot, never a retained run's files."""
        from .models import _snapshot_artifact
        from .errors import NativeBackendError

        if self._source is None or not self.needs_fresh_worker_snapshot(model, contract):
            raise NativeBackendError('no reusable model snapshot is prepared for worker launch')
        try:
            copied_bytes = sum((self._source.weights_dir / name).stat().st_size
                               for name in self._source.artifact_files)
        except OSError as error:
            raise NativeBackendError(f'could not inspect native model snapshot inputs: {error}') from error
        snapshot = _snapshot_artifact(self._source, contract, run_dir)
        self._entry = (self._entry[0], snapshot.weights_dir)
        self._prepared = None
        self._active = snapshot
        self.last = {'reused': False, 'copied_bytes': copied_bytes, 'snapshot_dir': str(snapshot.weights_dir)}
        return snapshot

    def clear(self):
        self._entry = None
        self._source = None
        self._prepared = None
        self._active = None
        self.last = None
