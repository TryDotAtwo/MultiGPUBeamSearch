"""Temporary tensor and collective stages for the Rust CNOT controller.

The calling thread and process group remain the caller's. These methods do
not own search iteration or history ordering. Their Torch tensor operations
are intentionally identical to the reference while the native CNOT stage
backend is being migrated.
"""
import torch
import torch.distributed as dist

from . import cnot_native as native
from . import cnot


class CnotStages:
    def __init__(self, world, root, identity, input_matrix, scorer, mode, audit):
        self.world, self.root, self.identity = world, root, identity
        self.input_matrix, self.scorer, self.mode, self.audit = input_matrix, scorer, mode, audit
        self.n, self.words = root.shape[1:]

    def initialize(self):
        world, root = self.world, self.root
        self.parents = root if world.rank == 0 else root[:0]
        self.lex = world.longs([0] if world.rank == 0 else [])
        owner = world.checked("initialize CNOT domain", lambda: native.cnot_owners(root, world.size).item())
        self.closed = root if owner == world.rank else root[:0]

    def goal(self, begin):
        if not self.parents.is_cuda:
            return cnot._goal(self.world, self.parents, self.lex, begin, self.identity)

        def local_goal(parents, lex, slot, identity):
            # Load and call inside _goal's error-coordination boundary.
            ops = native._cuda_ops()
            return ops.local_goal(parents, lex, slot, identity) if hasattr(ops, "local_goal") else None

        return cnot._goal(self.world, self.parents, self.lex, begin, self.identity,
                          local_goal=local_goal)

    def expand(self, begin, k):
        self.candidates, self.owners = self.world.checked("rank parent actions and expand",
            lambda: self._expand(begin, k))

    def _expand(self, begin, k):
        ops = native._cuda_ops() if self.parents.is_cuda else None
        if ops is None or not hasattr(ops, "expand_candidates"):
            # Retain CPU support and externally supplied older operator libraries.
            return cnot._expand(self.world, self.parents, self.lex, begin, self.scorer, self.mode, k)
        count = len(self.parents)
        alphabet = self.n * (self.n - 1)
        scores = (cnot._scores(self.scorer.parent_scores(self.parents, self.mode),
                               (count, alphabet), self.world.device)
                  if count and k < alphabet else None)
        return ops.expand_candidates(self.parents, self.lex, scores, begin, self.world.rank,
                                     self.world.size, k, self.mode == "proposal")

    def exchange_candidates(self):
        self.incoming = self.world.exchange(self.candidates, self.owners, native_packet=True)

    def deduplicate_and_score(self):
        self.children, self.meta, self.values = self.world.checked("deduplicate and evaluate distinct children",
            self._distinct)

    def _distinct(self):
        ops = native._cuda_ops() if self.incoming.is_cuda else None
        if ops is None or not hasattr(ops, "distinct_candidates"):
            return cnot._distinct(self.world, self.incoming, self.closed, self.n, self.words, self.scorer)
        children, meta = ops.distinct_candidates(self.incoming, self.closed, self.n, self.words)
        values = (cnot._scores(self.scorer.child_values(children), (len(children),), self.world.device)
                  if len(children) else torch.empty(0, dtype=torch.float64, device=self.world.device))
        return children, meta, values

    def prepare_selection(self):
        from .cnot_rust_selection import SelectionStages

        return SelectionStages(self.world, self.values, self.meta[:, 1])

    def commit_selection(self, result):
        self.ranks, self.local, self.new_lex = result

    def select(self, width):
        # Compatibility with the preceding controller extension, which calls
        # one Python selection stage rather than the Rust radix controller.
        self.ranks, self.local, self.new_lex = cnot.global_winners(self.world, self.values, self.meta[:, 1], width)

    def redistribute(self, depth):
        world, n, words = self.world, self.n, self.words
        count = len(self.local)
        ops = native._cuda_ops() if self.children.is_cuda else None
        # An externally supplied operator library may predate these optional
        # stage symbols. Keep its existing tensor implementation usable.
        use_native = ops is not None and all(hasattr(ops, name) for name in
                                             ("winner_packet", "sort_received", "frontier_views"))
        if use_native:
            self.closed, wire, targets = ops.winner_packet(self.ranks, self.local, self.new_lex,
                self.meta, self.children, self.closed, world.rank, world.size)
        else:
            slots = (self.ranks == world.rank).nonzero().flatten()
            kept = self.local[slots]
            owned_rows = self.children[kept]
            self.closed = torch.cat((self.closed, owned_rows))
            records = torch.stack((slots, self.new_lex[slots], self.meta[kept, 0],
                                   self.meta[kept, 3], self.ranks[slots]), 1)
            wire = torch.cat((records, owned_rows.reshape(len(kept), n * words)), 1)
            targets = slots * world.size // max(1, count)
        received = world.exchange(wire, targets, native_packet=True)
        received = ops.sort_received(received) if use_native else received[torch.argsort(received[:, 0])]
        stats = world.gather_fixed(world.longs([len(self.parents), len(self.candidates),
            (self.owners != world.rank).sum().item(), len(self.children), len(received)])).tolist()
        layer = dict(depth=depth, by_rank=stats, distinct=sum(s[3] for s in stats), retained=count)
        if self.audit:
            payload = torch.cat((self.values.to(torch.float64).contiguous().view(torch.int64)[:, None],
                                 self.meta, self.children.reshape(len(self.children), n * words)), 1)
            layer["candidates"] = world.gather_rows(payload).cpu().tolist()
            layer["winners"] = world.gather_rows(received).cpu().tolist()
        if not count:
            return count, layer, None
        if use_native:
            history_records, self.parents, self.lex = ops.frontier_views(received, n, words)
        else:
            history_records = received[:, 2:5].contiguous()
            self.parents = received[:, 5:].reshape(-1, n, words).contiguous()
            self.lex = received[:, 1].contiguous()
        return count, layer, history_records

    def history_edge(self, records, target, local_index):
        edge = self.world.longs([0, 0, 0])
        if target == self.world.rank:
            edge.copy_(records[local_index])
        if self.world.size > 1:
            dist.broadcast(edge, target)
        return tuple(edge.tolist())

    def verifies_construction(self, gates):
        return self.input_matrix.verifies_construction(gates)
