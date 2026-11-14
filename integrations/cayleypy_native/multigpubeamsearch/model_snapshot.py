"""Session-local reuse of a validated private MLP artifact snapshot."""
from pathlib import Path


class ModelSnapshotCache:
    """Retain one snapshot; callers serialize access using the session lock."""

    def __init__(self):
        self._entry = None
        self._source = None
        self._prepared = None
        self.last = None

    def prepare(self, model, contract, run_dir, cache_dir):
        from .models import _snapshot_artifact, _validate_artifact, verify_prepared_model
        from .errors import NativeBackendError

        key = (str(Path(cache_dir).resolve()), contract.graph_hash, model.backend, model.artifact_hash)
        self.last = None
        self._source = model
        self._prepared = None
        if self._entry is not None and self._entry[0] == key:
            # Keep only a path in the cache. A manifest returned to the user
            # can be mutable, so reconstruct the prepared model from disk.
            snapshot = _validate_artifact(self._entry[1], contract, model.backend)
            if snapshot.artifact_hash != model.artifact_hash:
                raise NativeBackendError("native model artifact changed before launch")
            # The source was validated before lookup. Recheck it after the
            # retained snapshot so a change during reuse is not hidden.
            verify_prepared_model(model, contract)
            self.last = {'reused': True, 'copied_bytes': 0, 'snapshot_dir': str(snapshot.weights_dir)}
            self._prepared = snapshot
            return snapshot
        try:
            copied_bytes = sum((model.weights_dir / name).stat().st_size for name in model.artifact_files)
        except OSError as error:
            raise NativeBackendError(f"could not inspect native model snapshot inputs: {error}") from error
        snapshot = _snapshot_artifact(model, contract, run_dir)
        self._entry = (key, snapshot.weights_dir)
        self.last = {'reused': False, 'copied_bytes': copied_bytes, 'snapshot_dir': str(snapshot.weights_dir)}
        return snapshot

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
        self.last = {'reused': False, 'copied_bytes': copied_bytes, 'snapshot_dir': str(snapshot.weights_dir)}
        return snapshot

    def clear(self):
        self._entry = None
        self._source = None
        self._prepared = None
        self.last = None
