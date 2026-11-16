"""Transitional tensor/collective batches for the Rust global cutoff."""
import torch


class SelectionStages:
    def __init__(self, world, values, word_keys):
        self.world, self.values, self.word_keys = world, values, word_keys

    def begin(self, width):
        count = int(self.world.sum(self.world.longs([len(self.values)])).item())
        self.selection_ops = None
        if self.values.is_cuda:
            from . import cnot_native as native

            ops = native._cuda_ops()
            if all(hasattr(ops, name) for name in
                   ("selection_prepare", "selection_records", "selection_order")):
                self.selection_ops = ops
                self.bits, self.selected, self.boundary, self.score_keys = ops.selection_prepare(
                    self.values, self.word_keys, count > width)
                return count
        scores = self.values.to(torch.float64) + 0.0
        self.bits = scores.contiguous().view(torch.int64)
        self.selected = torch.ones_like(self.word_keys, dtype=torch.bool)
        if count > width:
            self.selected.zero_()
            self.boundary = torch.ones_like(self.selected)
            self.score_keys = torch.where(self.bits < 0, ~self.bits, self.bits ^ -(1 << 63))
        return count

    def radix_pass(self, key, shift, remaining):
        keys = self.score_keys if key == 0 else self.word_keys
        if keys.is_cuda:
            from . import cnot_native as native

            ops = native._cuda_ops()
            if all(hasattr(ops, name) for name in ("radix_histogram", "radix_update")):
                digit, hist = ops.radix_histogram(keys, self.boundary, shift)
                self.world.sum(hist)
                return ops.radix_update(digit, hist, self.selected, self.boundary, remaining)
        digit = (keys >> shift) & 255
        hist = self.world.sum(torch.bincount(digit[self.boundary], minlength=256))
        prefix = torch.cumsum(hist, 0)
        cutoff = int(torch.searchsorted(prefix, remaining).item())
        below = int(prefix[cutoff - 1].item()) if cutoff else 0
        self.selected |= self.boundary & (digit < cutoff)
        self.boundary &= digit == cutoff
        return below, int(hist[cutoff].item())

    def accept_boundary(self):
        self.selected |= self.boundary

    def finish(self, count, width):
        if self.selection_ops is not None:
            records = self.selection_ops.selection_records(
                self.bits, self.word_keys, self.selected, self.world.rank)
        else:
            local = self.selected.nonzero().flatten()
            records = torch.stack((self.bits[local], self.word_keys[local],
                                   torch.full_like(local, self.world.rank), local), 1)
        records = self.world.gather_rows(records)
        if len(records) != min(count, width):
            raise RuntimeError("global radix cutoff returned an inconsistent frontier size")
        if self.selection_ops is not None:
            return self.selection_ops.selection_order(records)
        by_word = torch.argsort(records[:, 1], stable=True)
        in_order = by_word[torch.argsort(records[by_word, 0].contiguous().view(torch.float64), stable=True)]
        lexical_rank = torch.empty_like(by_word)
        lexical_rank[by_word] = torch.arange(len(by_word), device=self.world.device)
        return records[in_order, 2], records[in_order, 3], lexical_rank[in_order]
