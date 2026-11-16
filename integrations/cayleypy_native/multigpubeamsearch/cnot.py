"""Distributed beam search on packed, invertible GF(2) matrices."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import torch
import torch.distributed as dist

from .cnot_collectives import World
from . import cnot_native as native
from .cnot_selection import global_winners


@dataclass
class CnotResult:
    status: str
    gates: tuple[int, ...] | None  # construction from identity, in execution order
    layers: list[dict]
    path_routes: list[dict]  # retained edges in reduction order

    @property
    def length(self):
        return None if self.gates is None else len(self.gates)


@dataclass(frozen=True)
class _InputMatrix:
    n: int
    raw: bytes
    root: torch.Tensor
    rows: tuple[int, ...] | None = None
    rust_matrix: object | None = None

    def verifies_construction(self, gates):
        if self.rust_matrix is not None:
            return self.rust_matrix.verifies_construction(gates)
        return replay_cnot(self.n, gates) == self.rows


def pack_rows(rows, *, device=None):
    """Pack n Python row integers in little-endian 64-bit lanes."""
    n = len(rows)
    if not 2 <= n <= 256 or any(not isinstance(row, int) or not 0 <= row < 1 << n for row in rows):
        raise ValueError("expected 2..256 nonnegative n-bit row integers")
    words = (n + 63) // 64
    raw = b"".join(row.to_bytes(8 * words, "little") for row in rows)
    return torch.frombuffer(bytearray(raw), dtype=torch.int64).reshape(1, n, words).to(device=device)


def replay_cnot(n, gates):
    """Independent integer-row replay from identity; gate order is construction order."""
    rows = [1 << row for row in range(n)]
    for move in gates:
        if not 0 <= move < n * (n - 1):
            raise ValueError("physical CNOT ID outside alphabet")
        control, slot = divmod(move, n - 1)
        target = slot + (slot >= control)
        rows[target] ^= rows[control]
    return tuple(rows)


def _input(root, width, shortlist, depth, mode, validation="python"):
    if root.dtype != torch.int64 or root.ndim != 3 or root.shape[0] != 1:
        raise ValueError("expected one packed int64 matrix [1,n,ceil(n/64)]")
    n, words = root.shape[1:]
    if not 2 <= n <= 256 or words != (n + 63) // 64:
        raise ValueError("CNOT shape requires 2..256 qubits and ceil(n/64) words per row")
    if any(type(x) is not int for x in (width, shortlist, depth)) or not (
            1 <= width <= (2**63 - 1) // (n * (n - 1)) and shortlist >= 1 and depth >= 0):
        raise ValueError("invalid global width, shortlist, or depth")
    if mode not in ("qhat", "proposal"):
        raise ValueError("mode must be qhat or proposal")
    if validation not in ("python", "rust"):
        raise ValueError("validation must be python or rust")
    # Validate and search the same owned tensor. The caller may reuse or mutate
    # its tensor after admission; a contiguous() call alone can still alias it.
    snapshot = root.detach().clone(memory_format=torch.contiguous_format)
    raw = snapshot.cpu().numpy().tobytes()
    if validation == "rust":
        from .errors import NativeUnavailable
        from .rust_control import extension

        matrix_type = getattr(extension(), "CnotMatrix", None)
        if matrix_type is None:
            raise NativeUnavailable("Rust runtime lacks CNOT input validation; rebuild the matching PyO3 module")
        return _InputMatrix(n, raw, snapshot, rust_matrix=matrix_type(n, raw))
    rows = tuple(int.from_bytes(raw[i:i + 8 * words], "little") for i in range(0, len(raw), 8 * words))
    if any(row.bit_length() > n for row in rows):
        raise ValueError("nonzero CNOT row padding")
    pivots = {}
    for row in rows:
        while row and row.bit_length() in pivots:
            row ^= pivots[row.bit_length()]
        if not row:
            raise ValueError("CNOT matrix is singular")
        pivots[row.bit_length()] = row
    return _InputMatrix(n, raw, snapshot, rows=rows)


def _goal(world, parents, lex, begin, identity, *, local_goal=None):
    key, slot = 2**63 - 1, -1
    error = None
    try:
        n = parents.size(1)
        alphabet = n * (n - 1)
        result = local_goal(parents, lex, begin, identity) if len(parents) and local_goal else None
        if result is not None:
            key, slot = result
        elif len(parents):
            diff = parents ^ identity
            changed = diff.ne(0).any(2)
            target = changed.long().argmax(1)
            index = torch.arange(len(parents), device=world.device)
            needed = diff[index, target]
            matches = (parents == needed[:, None]).all(2)
            matches[index, target] = False
            hits = ((changed.sum(1) == 1) & matches.any(1)).nonzero().flatten()
            if len(hits):
                hit = hits[lex[hits].argmin()]
                control = matches[hit].long().argmax()
                move = control * (n - 1) + target[hit] - (target[hit] > control).long()
                key, slot = int((lex[hit] * alphabet + move).item()), begin + int(hit.item())
    except Exception as exc:
        if world.size == 1:
            raise
        error = f"rank {world.rank}: {type(exc).__name__}: {exc}"
    # Use the existing goal exchange to coordinate failures: no extra
    # collective or host read is required on the successful path.
    gathered = world.gather_fixed(world.longs([key, slot, int(error is not None)])).tolist()
    if any(failed for _, _, failed in gathered):
        errors = [None] * world.size
        dist.all_gather_object(errors, error)
        raise RuntimeError("find CNOT goal failed: " + " | ".join(e for e in errors if e))
    best_key, best_slot = min((value, index) for value, index, _ in gathered)
    return None if best_slot < 0 else (best_slot, best_key % alphabet)


def _scores(tensor, shape, device):
    if tensor.shape != shape or tensor.device != device or not tensor.is_floating_point():
        raise ValueError(f"expected floating scores on {device} with shape {shape}")
    if not torch.isfinite(tensor).all().item():
        raise ValueError("nonfinite CNOT model score")
    return tensor


def _expand(world, parents, lex, begin, scorer, mode, k):
    n, words = parents.shape[1:]
    count, alphabet = len(parents), n * (n - 1)
    all_ids = torch.arange(alphabet, device=world.device)
    if count and k < alphabet:
        scores = _scores(scorer.parent_scores(parents, mode), (count, alphabet), world.device)
        moves = torch.argsort(scores, dim=1, descending=mode == "proposal", stable=True)[:, :k]
    else:
        moves = all_ids[:k].expand(count, k)
    indices = torch.arange(count, device=world.device).repeat_interleave(k)
    moves = moves.reshape(-1)
    children = native.cnot_children(parents, indices, moves)
    parent_slots = indices + begin
    generation = parent_slots * alphabet + torch.arange(k, device=world.device).repeat(count)
    meta = torch.stack((parent_slots, lex[indices] * alphabet + moves,
                        generation, native.pack_cnot_route(world.rank, moves)), 1)
    return torch.cat((meta, children.reshape(-1, n * words)), 1), native.cnot_owners(children, world.size)


def _distinct(world, incoming, closed, n, words, scorer):
    incoming = incoming[torch.argsort(incoming[:, 2], stable=True)]
    children = incoming[:, 4:].reshape(-1, n, words).contiguous()
    first = native.cnot_first_unseen(closed, children)
    children, meta = children[first], incoming[first, :4]
    values = (_scores(scorer.child_values(children), (len(children),), world.device)
              if len(children) else torch.empty(0, dtype=torch.float64, device=world.device))
    return children, meta, values


def _reconstruct(world, history, final, input_matrix):
    slot, last_move = final
    moves, routes = [last_move], []
    for count, previous_count, records in reversed(history):
        target = slot * world.size // count
        edge = world.longs([0, 0, 0])
        if target == world.rank:
            begin = (target * count + world.size - 1) // world.size
            edge.copy_(records[slot - begin])
        if world.size > 1:
            dist.broadcast(edge, target)
        slot, route, owner = edge.tolist()
        source, move = divmod(route, 1 << 16)
        if source != slot * world.size // previous_count:
            raise RuntimeError("CNOT history disagrees with parent world rank")
        moves.append(move)
        routes.append(dict(source=source, hash_owner=owner, target=target, move=move))
    gates = tuple(moves)  # backtracking reversed the reduction, including its terminal gate
    if slot != 0 or not input_matrix.verifies_construction(gates):
        raise RuntimeError("CNOT construction failed bitwise replay from the identity")
    return gates, routes[::-1]


def _search_engine(engine, max_depth):
    if engine not in ("auto", "python", "rust"):
        raise ValueError("CNOT engine must be auto, python, or rust")
    if engine == "python":
        return "python", None
    from .errors import NativeUnavailable
    from .rust_control import extension

    if max_depth > 2**64 - 1:
        if engine == "auto":
            return "python", None
        raise ValueError("CNOT Rust engine depth exceeds u64")
    try:
        run = getattr(extension(), "run_cnot", None)
    except NativeUnavailable:
        if engine == "rust":
            raise
        return "python", None
    if run is None:
        if engine == "rust":
            raise NativeUnavailable("Rust runtime lacks CNOT search; rebuild the matching PyO3 module")
        return "python", None
    return "rust", run


def _native_stages_runner():
    """Optional Rust-owned stage backend; older PyO3 modules remain usable."""
    from .rust_control import extension

    return getattr(extension(), "run_cnot_native", None)


def _admit(root, scorer, width, shortlist, max_depth, mode, audit, validation, engine):
    input_matrix = _input(root, width, shortlist, max_depth, mode, validation)
    engine, runner = _search_engine(engine, max_depth)
    metadata = getattr(scorer, "metadata", {})
    fingerprints = ("upstream_commit", "checkpoint_sha256", "feature_registry_sha256",
                    "training_manifest_sha256", "training_result_sha256")
    config = (width, shortlist, max_depth, mode, audit, validation, engine, tuple(input_matrix.root.shape),
              tuple(metadata.get(key) for key in fingerprints))
    hasher = hashlib.sha256(input_matrix.raw)
    hasher.update(json.dumps(config).encode())
    return input_matrix, hasher.digest(), runner


@torch.inference_mode(False)
@torch.no_grad()
def search_cnot(root, scorer, *, width, shortlist, max_depth, mode="qhat", audit=False,
                validation="python", engine="auto"):
    """Run collectively on the initialized world, or directly on one CPU/GPU.

    Every rank supplies the same root, limits and weights. Each rank binds its
    own scorer to its local GPU. Audit records full candidate/winner matrices
    only when requested for small acceptance runs. ``validation="rust"``
    selects the optional Rust input and construction-path validator. The
    default engine uses the Rust search controller when installed; ``python``
    retains the reference controller and ``rust`` requires the extension.
    Tensor operations, scorer callbacks and collectives are unchanged.
    """
    with World(root.device) as world:
        input_matrix, digest, runner = world.checked("validate CNOT input", lambda: _admit(
            root, scorer, width, shortlist, max_depth, mode, audit, validation, engine))
        root = input_matrix.root
        identities = world.gather_fixed(world.longs(list(digest)))
        if not (identities == identities[0]).all().item():
            raise ValueError("CNOT root, search settings or scorer fingerprints differ between ranks")
        world.start_transport(prefer_rust=runner is not None)
        identity = native.cnot_identity(input_matrix.n, device=world.device)
        if torch.equal(root, identity):
            return CnotResult("solved", (), [], [])
        if runner is not None:
            native_runner = _native_stages_runner()
            use_native = world.sum(world.longs([int(native_runner is not None)])).item() == world.size
            alphabet = input_matrix.n * (input_matrix.n - 1)
            if use_native:
                return native_runner(world, root, identity, input_matrix, scorer, mode, audit,
                                     width, min(shortlist, alphabet), alphabet, max_depth)
            from .cnot_rust_stages import CnotStages

            stages = CnotStages(world, root, identity, input_matrix, scorer, mode, audit)
            return runner(stages, world.rank, world.size, width, min(shortlist, alphabet), alphabet, max_depth)
        return _search(world, root, identity, input_matrix, scorer, width, shortlist, max_depth, mode, audit)


def _search(world, root, identity, input_matrix, scorer, width, shortlist, max_depth, mode, audit):
    n, words = root.shape[1:]
    k = min(shortlist, n * (n - 1))
    parents = root if world.rank == 0 else root[:0]
    lex = world.longs([0] if world.rank == 0 else [])
    owner = world.checked("initialize CNOT domain", lambda: native.cnot_owners(root, world.size).item())
    closed = root if owner == world.rank else root[:0]
    begin, previous_count = 0, 1
    history, layers = [], []
    for step in range(max_depth):
        final = _goal(world, parents, lex, begin, identity)
        if final is not None:
            gates, routes = _reconstruct(world, history, final, input_matrix)
            return CnotResult("solved", gates, layers, routes)
        if step + 1 == max_depth:
            break  # every one-gate completion above was tested over the whole alphabet
        candidates, owners = world.checked("rank parent actions and expand",
            lambda: _expand(world, parents, lex, begin, scorer, mode, k))
        incoming = world.exchange(candidates, owners)
        children, meta, values = world.checked("deduplicate and evaluate distinct children",
            lambda: _distinct(world, incoming, closed, n, words, scorer))
        ranks, local, new_lex = global_winners(world, values, meta[:, 1], width)
        count = len(local)
        slots = (ranks == world.rank).nonzero().flatten()
        kept = local[slots]
        owned_rows = children[kept]
        closed = torch.cat((closed, owned_rows))
        records = torch.stack((slots, new_lex[slots], meta[kept, 0], meta[kept, 3], ranks[slots]), 1)
        wire = torch.cat((records, owned_rows.reshape(len(kept), n * words)), 1)
        targets = slots * world.size // max(1, count)
        received = world.exchange(wire, targets)
        received = received[torch.argsort(received[:, 0])]
        stats = world.gather_fixed(world.longs([len(parents), len(candidates),
            (owners != world.rank).sum().item(), len(children), len(received)])).tolist()
        layer = dict(depth=step + 1, by_rank=stats, distinct=sum(s[3] for s in stats), retained=count)
        if audit:
            payload = torch.cat((values.to(torch.float64).contiguous().view(torch.int64)[:, None],
                                 meta, children.reshape(len(children), n * words)), 1)
            layer["candidates"] = world.gather_rows(payload).cpu().tolist()
            layer["winners"] = world.gather_rows(received).cpu().tolist()
        layers.append(layer)
        if not count:
            return CnotResult("exhausted", None, layers, [])
        begin = (world.rank * count + world.size - 1) // world.size
        history.append((count, previous_count, received[:, 2:5].contiguous()))
        previous_count = count
        parents = received[:, 5:].reshape(-1, n, words).contiguous()
        lex = received[:, 1].contiguous()
    return CnotResult("depth_limit", None, layers, [])
