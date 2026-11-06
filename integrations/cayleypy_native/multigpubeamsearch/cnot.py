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


def _input(root, width, shortlist, depth, mode):
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
    raw = root.detach().contiguous().cpu().numpy().tobytes()
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
    return rows, raw


def _goal(world, parents, lex, begin, identity):
    n = parents.size(1)
    alphabet = n * (n - 1)
    key, slot = 2**63 - 1, -1
    if len(parents):
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
    best_key, best_slot = min(world.gather_fixed(world.longs([key, slot])).tolist())
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


def _reconstruct(world, history, final, input_rows):
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
    if slot != 0 or replay_cnot(len(input_rows), gates) != input_rows:
        raise RuntimeError("CNOT construction failed bitwise replay from the identity")
    return gates, routes[::-1]


@torch.inference_mode(False)
@torch.no_grad()
def search_cnot(root, scorer, *, width, shortlist, max_depth, mode="qhat", audit=False):
    """Run collectively on the initialized world, or directly on one CPU/GPU.

    Every rank supplies the same root, limits and weights. Each rank binds its
    own scorer to its local GPU. Audit records full candidate/winner matrices
    only when requested for small acceptance runs.
    """
    with World(root.device) as world:
        input_rows, raw = world.checked("validate CNOT input",
                                        lambda: _input(root, width, shortlist, max_depth, mode))
        metadata = getattr(scorer, "metadata", {})
        config = (width, shortlist, max_depth, mode, audit, tuple(root.shape),
                  metadata.get("checkpoint_sha256"), metadata.get("feature_registry_sha256"))
        digest = hashlib.sha256(raw + json.dumps(config).encode()).digest()
        identities = world.gather_fixed(world.longs(list(digest)))
        if not (identities == identities[0]).all().item():
            raise ValueError("CNOT root, search settings or scorer fingerprints differ between ranks")
        identity = native.cnot_identity(len(input_rows), device=world.device)
        if torch.equal(root, identity):
            return CnotResult("solved", (), [], [])
        return _search(world, root, identity, input_rows, scorer, width, shortlist, max_depth, mode, audit)


def _search(world, root, identity, input_rows, scorer, width, shortlist, max_depth, mode, audit):
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
            gates, routes = _reconstruct(world, history, final, input_rows)
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
