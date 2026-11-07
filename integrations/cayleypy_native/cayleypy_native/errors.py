"""Capability rejection is distinct from a failure of a selected native backend."""


class NativeUnavailable(RuntimeError):
    """Unsupported configuration, detected before native execution begins."""


class NativeBackendError(RuntimeError):
    """A selected native backend or its artifacts failed; never silently fall back."""


class NativeFallbackWarning(UserWarning):
    """An auto-dispatched call uses CayleyPy's original search instead."""
