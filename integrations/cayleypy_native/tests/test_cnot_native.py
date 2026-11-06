import pytest
import torch

from multigpubeamsearch.cnot_native import (
    cnot_children, cnot_first_unseen, cnot_identity, cnot_is_identity, cnot_owners,
    pack_cnot_route, unpack_cnot_route,
)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA domain acceptance requires a CUDA device"))])
def device(request):
    return torch.device(request.param)


def pack_dense(states):
    """Independent packing oracle, including the signed high bit of each int64."""
    batch, n, _ = states.shape
    words = torch.zeros((batch, n, (n + 63) // 64), dtype=torch.int64)
    for column in range(n):
        words[:, :, column // 64] |= states[:, :, column].to(torch.int64) << (column % 64)
    return words


@pytest.mark.parametrize("n", [3, 63, 64, 65, 127, 128, 129, 144, 255, 256])
def test_selected_children_against_dense_oracle(n, device):
    source = torch.randint(2, (3, n, n), generator=torch.Generator().manual_seed(n), dtype=torch.uint8)
    packed = pack_dense(source)
    # Deliberately contaminate padding in every row. No logical bit may change.
    if n % 64:
        packed[..., -1] |= -1 << (n % 64)
    parents = packed.to(device)
    original = parents.clone()
    gates = [(n - 1, n - 2), (0, n - 1), (n // 2, 0), (n // 2, n // 2 + 1), (1, 0)]
    parent_indices = [2, 0, 1, 2, 2]
    moves = [c * (n - 1) + t - int(t > c) for c, t in gates]
    expected = source[parent_indices].clone()
    for child, (c, t) in enumerate(gates):
        expected[child, t] ^= expected[child, c]
    result = cnot_children(parents, torch.tensor(parent_indices, device=device),
                           torch.tensor(moves, dtype=torch.uint16, device=device))
    assert torch.equal(result.cpu(), pack_dense(expected))
    assert torch.equal(parents, original)
    assert result.is_contiguous()
    assert torch.equal(cnot_owners(result, 250)[0:4].cpu(), cnot_owners(result.cpu(), 250)[0:4])
    assert torch.equal(cnot_owners(parents, 251).cpu(), cnot_owners(pack_dense(source), 251))


def test_physical_history_replays_noncommuting_high_moves(device):
    identity = cnot_identity(144, device=device)
    assert torch.equal(identity.cpu(), pack_dense(torch.eye(144, dtype=torch.uint8)[None]))
    # 143 -> 142, then 142 -> 0. Replaying in the opposite order changes row 0.
    forward = [20591, 142 * 143]
    state = identity
    history = []
    for source, physical in zip([65535, 257], forward):
        move = torch.tensor([physical], dtype=torch.int64, device=device)
        history.append(pack_cnot_route(source, move))
        state = cnot_children(state, torch.tensor([0], device=device), move)
    assert state[0, 0, -1].item() == (1 << 14) | (1 << 15)
    assert not cnot_is_identity(state).item()
    ranks, moves = unpack_cnot_route(torch.cat(history))
    assert ranks.tolist() == [65535, 257]
    assert moves.tolist() == forward
    for move in moves.flip(0):
        state = cnot_children(state, torch.tensor([0], device=device), move.reshape(1).to(torch.uint16))
    assert cnot_is_identity(state).item() and torch.equal(state, identity)


def test_every_144_qubit_move_in_bounded_batches(device):
    n = 144
    identity = cnot_identity(n, device=device)
    pairs = torch.cartesian_prod(torch.arange(n), torch.arange(n))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]].to(device)
    for begin in range(0, len(pairs), 257):
        control, target = pairs[begin:begin + 257].unbind(dim=1)
        local = torch.arange(len(target), device=device)
        expected = identity.expand(len(target), -1, -1).clone()
        expected[local, target, control // 64] ^= 1 << (control % 64)
        children = cnot_children(identity, torch.zeros_like(local), (begin + local).to(torch.uint16))
        assert torch.equal(children, expected)


def test_dedup_owner_agrees_across_devices_and_tail_padding(device):
    n = 70
    identity = cnot_identity(n)
    move = torch.arange(80)
    states = cnot_children(identity, torch.zeros_like(move), move)
    states = torch.cat([states, states[:7], identity])
    dirty = states.clone()
    dirty[..., -1] |= -64
    reference = cnot_owners(states, 17)
    assert reference.unique().numel() > 8
    assert torch.equal(cnot_owners(dirty.to(device), 17).cpu(), reference)
    assert torch.equal(reference[:7], reference[-8:-1])
    assert cnot_is_identity(dirty.to(device)).nonzero().flatten().tolist() == [states.size(0) - 1]


def test_empty_alphabet_and_empty_batch(device):
    one = cnot_identity(1, device=device)
    ids = torch.empty(0, device=device, dtype=torch.int64)
    assert cnot_is_identity(one).item()
    assert cnot_children(one[:0], ids, ids).shape == (0, 1, 1)
    assert cnot_owners(one[:0], 3).shape == (0,)
    assert cnot_first_unseen(one, one[:0]).shape == (0,)
    assert cnot_first_unseen(one, one).numel() == 0


def test_full_row_unique_keeps_first_generation_and_removes_closed(device):
    # All states agree through the final row. Some carry dirt outside n bits.
    n = 144
    states = cnot_identity(n).expand(9, -1, -1).clone()
    states[:, -1, -1] |= torch.tensor([0, 4, 8, 16, 32, 64, 128, 256, 512])
    closed = states[[0, 2, 0, 7]]
    children = states[[2, 1, 3, 1, 0, 3, 7, 5, 8, 8, 6]]
    children[3, :, -1] |= -65536
    original = children.clone()
    assert cnot_first_unseen(closed.to(device), children.to(device)).tolist() == [1, 2, 7, 8, 10]
    assert cnot_first_unseen(closed[:0].to(device), children.to(device)).tolist() == [0, 1, 2, 4, 6, 7, 8, 10]
    assert torch.equal(original, children)


def test_unique_across_multiple_sort_tiles(device):
    generator = torch.Generator().manual_seed(4105)
    pool = torch.randint(512, (400, 9, 1), generator=generator)
    pool[:, 0, 0] = torch.arange(400)
    prior = torch.randint(400, (173,), generator=generator)
    candidates = torch.randint(400, (3793,), generator=generator)
    seen = set(prior.tolist())
    expected = []
    for i, candidate in enumerate(candidates.tolist()):
        if candidate not in seen:
            expected.append(i)
            seen.add(candidate)
    assert cnot_first_unseen(pool[prior].to(device), pool[candidates].to(device)).tolist() == expected


def test_invalid_descriptors_and_routes():
    p = cnot_identity(144)
    for bad in [-1, 20592, 2**16 + 7]:
        with pytest.raises(ValueError, match="outside the physical alphabet"):
            cnot_children(p, torch.tensor([0]), torch.tensor([bad]))
    with pytest.raises((RuntimeError, IndexError)):
        cnot_children(p, torch.tensor([-1]), torch.tensor([0]))
    with pytest.raises(RuntimeError, match="fit uint16"):
        pack_cnot_route(0, torch.tensor([65536]))
    with pytest.raises(ValueError, match="source rank"):
        pack_cnot_route(65536, torch.tensor([9]))
