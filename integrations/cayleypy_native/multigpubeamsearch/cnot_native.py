"""Packed GF(2) domain primitives. Only the selected physical moves are materialized."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _cuda_ops():
    # Cluster images can ship the matching prebuilt PyTorch operator library.
    # Keep the CUDA compiler and Ninja optional on those compute nodes.
    library = os.environ.get("MGBS_CNOT_DOMAIN_LIBRARY")
    if library:
        torch.ops.load_library(library)
        return torch.ops.multigpubeamsearch_cnot_domain

    from torch.utils.cpp_extension import load

    package = Path(__file__).resolve().parent
    root = package / "_native_source"
    if not (root / "cuda/cnot_domain.cu").is_file():
        root = package.parents[2]
    sources = [root / "cuda/cnot_torch.cpp", root / "cuda/cnot_domain.cu", root / "cuda/cnot_unique.cu"]
    headers = ["cuda/cnot_domain.hpp", "src/cnot_domain.hpp", "src/hash.hpp",
               "src/types.hpp", "src/config.hpp"]
    digest = hashlib.sha256()
    for path in [*sources, *(root / header for header in headers)]:
        digest.update(path.read_bytes())
    # hash.hpp shares the original permutation ABI, whose build constants are
    # required by its headers. CNOT matrix dimensions are independently dynamic.
    definitions = ["BEAM_STATE_LOGICAL_BYTES=120", "BEAM_STATE_PHYSICAL_BYTES=128",
                   "BEAM_STATE_ALIGNMENT=16", "BEAM_MOVE_COUNT=1"]
    define_flags = ["-D" + definition for definition in definitions]
    load(name="beam_cnot_domain_" + digest.hexdigest()[:16],
         sources=[str(source) for source in sources],
         extra_cflags=(["/O2"] if os.name == "nt" else ["-O2"]) + define_flags,
         extra_cuda_cflags=["-O2", *define_flags], is_python_module=False)
    return torch.ops.multigpubeamsearch_cnot_domain


def _check_states(packed):
    if packed.dtype != torch.int64 or packed.ndim != 3:
        raise ValueError("packed CNOT matrices require int64 [batch,n,ceil(n/64)]")
    n = packed.size(1)
    if not 1 <= n <= 256 or packed.size(2) != (n + 63) // 64:
        raise ValueError("packed CNOT matrices require 1..256 qubits and ceil(n/64) words per row")
    if packed.device.type not in ("cpu", "cuda"):
        raise ValueError("CNOT matrices require a CUDA or CPU device")
    return n


def cnot_identity(n, *, device=None):
    """Return the packed identity as one contiguous matrix [1,n,ceil(n/64)]."""
    if not 1 <= n <= 256:
        raise ValueError("CNOT qubit count must be in [1, 256]")
    rows = torch.arange(n, device=device, dtype=torch.int64)
    identity = torch.zeros((1, n, (n + 63) // 64), device=rows.device, dtype=torch.int64)
    identity[0, rows, rows // 64] = 1 << (rows % 64)
    return identity


def cnot_is_identity(packed):
    """Compare every logical bit of each state with the identity."""
    n = _check_states(packed)
    identity = cnot_identity(n, device=packed.device)
    if n % 64:
        prefix = (packed[..., :-1] == identity[..., :-1]).all(dim=(1, 2))
        last = packed[..., -1] & ((1 << (n % 64)) - 1)
        return prefix & (last == identity[..., -1]).all(dim=1)
    return (packed == identity).all(dim=(1, 2))


def cnot_children(parents, parent_indices, moves):
    """Gather K parents and apply the K physical CNOTs, leaving parents immutable.

    Inputs are on one device. Output rows have zero tail padding. Descriptors
    are one-dimensional, indices int64 and moves uint16 or int64. An invalid
    descriptor raises on CPU and triggers a device error on CUDA.
    """
    n = _check_states(parents)
    if parent_indices.device != parents.device or moves.device != parents.device:
        raise ValueError("CNOT parents and descriptors must use the same device")
    if (parent_indices.dtype != torch.int64 or parent_indices.ndim != 1 or moves.ndim != 1
            or moves.shape != parent_indices.shape or moves.dtype not in (torch.uint16, torch.int64)):
        raise ValueError("expected equally sized int64 parent indices and uint16/int64 moves")
    if parents.is_cuda:
        return _cuda_ops().children(parents.contiguous(), parent_indices.contiguous(), moves.contiguous())
    descriptors = moves.to(torch.int64)
    children = parents.index_select(0, parent_indices)
    if not moves.numel():
        return children
    if n == 1 or int(descriptors.min()) < 0 or int(descriptors.max()) >= n * (n - 1):
        raise ValueError("CNOT move is outside the physical alphabet")
    controls = descriptors // (n - 1)
    offsets = descriptors % (n - 1)
    targets = offsets + (offsets >= controls)
    child = torch.arange(moves.numel(), device=parents.device)
    children[child, targets] ^= children[child, controls]
    if n % 64:
        children[..., -1] &= (1 << (n % 64)) - 1
    return children


def pack_cnot_route(source_rank, moves):
    """Zero-extend the 32-bit CNOT route into a collective-friendly int64 lane."""
    if not isinstance(source_rank, int) or not 0 <= source_rank < 65536:
        raise ValueError("CNOT source rank must fit uint16")
    if moves.dtype not in (torch.int64, torch.uint16):
        raise ValueError("CNOT moves must be uint16 or int64")
    if moves.dtype == torch.int64:
        torch._assert_async(((moves >= 0) & (moves <= 65535)).all(), "CNOT move must fit uint16")
    return moves.to(torch.int64) | (source_rank << 16)


def unpack_cnot_route(routes):
    """Return int64 source ranks and physical move IDs from zero-extended routes."""
    if routes.dtype != torch.int64:
        raise ValueError("CNOT route lanes must be int64")
    torch._assert_async(((routes >= 0) & (routes <= 0xffffffff)).all(), "CNOT route must fit uint32")
    return routes >> 16, routes & 0xffff


def _mix64(value):
    import numpy as np

    value = (value ^ (value >> 30)) * np.uint64(0xbf58476d1ce4e5b9)
    value = (value ^ (value >> 27)) * np.uint64(0x94d049bb133111eb)
    return value ^ (value >> 31)


def cnot_owners(packed, world_size):
    """Assign each full matrix to one rank. Owners never establish equality."""
    n = _check_states(packed)
    if not isinstance(world_size, int) or not 1 <= world_size <= 65536:
        raise ValueError("CNOT world size must be in [1, 65536]")
    if packed.is_cuda:
        return _cuda_ops().owners(packed.contiguous(), world_size)

    import numpy as np

    # Independent CPU reference for the CUDA position hash and the native
    # hash128_owner_distribution_key. 'uint64' operations wrap modulo 2**64.
    words = packed.contiguous().numpy().view(np.uint64)
    if n % 64:
        words = words.copy()
        words[..., -1] &= np.uint64((1 << (n % 64)) - 1)
    stride = n * packed.size(2)
    words = words.reshape(packed.size(0), stride)
    position = np.arange(1, stride + 1, dtype=np.uint64)
    lo = np.bitwise_xor.reduce(_mix64(words ^ (position * np.uint64(0x9e3779b97f4a7c15))), axis=1)
    hi = np.bitwise_xor.reduce(_mix64(words + position * np.uint64(0xd6e8feb86659fd93)), axis=1)
    lo ^= np.uint64(n)
    hi ^= np.uint64(n << 32)
    mixed = lo ^ ((hi << 32) | (hi >> 32)) ^ np.uint64(0x243f6a8885a308d3)
    mixed ^= _mix64(hi + np.uint64(0x9e3779b97f4a7c15))
    owners = _mix64(mixed) % np.uint64(world_size)
    return torch.from_numpy(owners.astype(np.int64))


def cnot_first_unseen(closed, children):
    """First child index per previously unseen full matrix, in original generation order."""
    n = _check_states(closed)
    if _check_states(children) != n or closed.device != children.device:
        raise ValueError("closed and child CNOT matrices must share n and device")
    if children.is_cuda:
        return _cuda_ops().first_unseen(closed.contiguous(), children.contiguous())
    if not children.size(0):
        return torch.empty(0, dtype=torch.int64)
    states = torch.cat((closed, children))
    if n % 64:
        states[..., -1] &= (1 << (n % 64)) - 1
    _, groups = torch.unique(states.flatten(1), dim=0, return_inverse=True)
    ordinals = torch.arange(states.size(0), dtype=torch.int64)
    first = torch.full_like(ordinals, states.size(0))
    first.scatter_reduce_(0, groups, ordinals, reduce="amin")
    child_groups = groups[closed.size(0):]
    return (first[child_groups] == ordinals[closed.size(0):]).nonzero().flatten()
