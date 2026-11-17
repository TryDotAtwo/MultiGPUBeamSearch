"""Public dispatch prefers the Rust owner when the matching worker exists."""
from multigpubeamsearch import dispatch, persistent
from multigpubeamsearch.persistent import NativeSession, current_session
from multigpubeamsearch.options import NativeOptions


def test_default_session_selects_rust_when_runtime_is_available(monkeypatch):
    monkeypatch.setattr(persistent, "_rust_resident_available", lambda: True, raising=False)
    session = NativeSession()
    try:
        assert session._engine == "rust"
    finally:
        session.close()


def test_default_session_preserves_python_fallback_without_runtime(monkeypatch):
    monkeypatch.setattr(persistent, "_rust_resident_available", lambda: False, raising=False)
    session = NativeSession()
    try:
        assert session._engine == "python"
    finally:
        session.close()


def test_unscoped_native_dispatch_owns_a_temporary_rust_session(monkeypatch):
    monkeypatch.setattr(dispatch, "_rust_resident_available", lambda: True, raising=False)
    seen = []

    def search(*args):
        session = current_session()
        seen.append(session)
        return "searched"

    monkeypatch.setattr(dispatch, "_search_impl", search)
    result = dispatch._search(None, object(), {}, NativeOptions(), "native")

    assert result == "searched"
    assert len(seen) == 1
    assert seen[0] is not None and seen[0]._engine == "rust"
    assert seen[0].closed
    assert current_session() is None


def test_explicit_python_session_keeps_legacy_owner(monkeypatch):
    monkeypatch.setattr(
        dispatch,
        "_rust_resident_available",
        lambda: (_ for _ in ()).throw(AssertionError("explicit session must be preserved")),
        raising=False,
    )
    seen = []
    monkeypatch.setattr(
        dispatch,
        "_search_impl",
        lambda *args: seen.append(current_session()) or "searched",
    )
    with NativeSession(engine="python") as session:
        assert dispatch._search(None, object(), {}, NativeOptions(), "native") == "searched"
        assert seen == [session]

