"""Exact global cutoff for a raw scalar value and a physical-word tie break."""
import torch


def global_winners(world, values, word_keys, width):
    """Return (owner rank, owner's candidate index, new lex rank) in beam order.

    Reduce 256 counters per radix pass. Every owner retains the entire unresolved
    boundary. Replicate only the final B metadata rows. No per-shard beam cap.
    """
    count = int(world.sum(world.longs([len(values)])).item())
    scores = values.to(torch.float64) + 0.0  # Give signed zero one sortable key.
    bits = scores.contiguous().view(torch.int64)
    selected = torch.ones_like(word_keys, dtype=torch.bool)
    if count > width:
        selected.zero_()
        boundary = torch.ones_like(selected)
        remaining = width
        score_keys = torch.where(bits < 0, ~bits, bits ^ -(1 << 63))
        for key in (score_keys, word_keys):
            for shift in range(56, -1, -8):
                digit = (key >> shift) & 255
                hist = world.sum(torch.bincount(digit[boundary], minlength=256))
                prefix = torch.cumsum(hist, 0)
                cutoff = int(torch.searchsorted(prefix, remaining).item())
                below = int(prefix[cutoff - 1].item()) if cutoff else 0
                selected |= boundary & (digit < cutoff)
                boundary &= digit == cutoff
                remaining -= below
                if remaining == int(hist[cutoff].item()):
                    selected |= boundary
                    break
            else:
                continue
            break
    local = selected.nonzero().flatten()
    records = torch.stack((bits[local], word_keys[local], torch.full_like(local, world.rank), local), 1)
    records = world.gather_rows(records)
    if len(records) != min(count, width):
        raise RuntimeError("global radix cutoff returned an inconsistent frontier size")
    by_word = torch.argsort(records[:, 1], stable=True)
    in_order = by_word[torch.argsort(records[by_word, 0].contiguous().view(torch.float64), stable=True)]
    lexical_rank = torch.empty_like(by_word)
    lexical_rank[by_word] = torch.arange(len(by_word), device=world.device)
    return records[in_order, 2], records[in_order, 3], lexical_rank[in_order]
