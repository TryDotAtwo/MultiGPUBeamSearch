import numpy as np


def pair_pack_ff1_activations(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Interleave two MxK activation matrices into Mx(2K)."""
    assert x0.shape == x1.shape
    packed = np.empty((x0.shape[0], x0.shape[1] * 2), dtype=x0.dtype)
    packed[:, 0::2] = x0
    packed[:, 1::2] = x1
    return packed


def pair_pack_ff1_weights(w0: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """Build the exact 2:4 block-diagonal sparse weight matrix.

    Inputs use the Transformer convention KxN. The result is (2K)x(2N):
    output columns [0,N) consume only even activation lanes, while [N,2N)
    consume only odd lanes. Every consecutive four values along K contain
    exactly two structural nonzeros.
    """
    assert w0.shape == w1.shape
    k, n = w0.shape
    packed = np.zeros((2 * k, 2 * n), dtype=w0.dtype)
    packed[0::2, :n] = w0
    packed[1::2, n:] = w1
    return packed


def test_pair_packed_ff1_matches_two_independent_dense_products() -> None:
    rng = np.random.default_rng(20260902)
    x0 = rng.standard_normal((13, 256), dtype=np.float32)
    x1 = rng.standard_normal((13, 256), dtype=np.float32)
    w0 = rng.standard_normal((256, 1024), dtype=np.float32)
    w1 = rng.standard_normal((256, 1024), dtype=np.float32)

    actual = pair_pack_ff1_activations(x0, x1) @ pair_pack_ff1_weights(w0, w1)
    expected = np.concatenate((x0 @ w0, x1 @ w1), axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_pair_packed_ff1_has_exact_two_of_four_structure() -> None:
    # Non-zero sentinels make the structural check independent of random zeros.
    w0 = np.ones((256, 1024), dtype=np.float32)
    w1 = np.ones((256, 1024), dtype=np.float32)
    packed = pair_pack_ff1_weights(w0, w1)
    groups = packed.reshape(128, 4, 2048)
    assert np.all(np.count_nonzero(groups, axis=1) == 2)


def test_pair_packed_ff1_useful_flop_accounting() -> None:
    m = 32768
    original_pair = 2 * (2 * m * 256 * 1024)
    physical_dense_shape = 2 * 2048 * m * 512
    assert physical_dense_shape == 2 * original_pair


def test_transposed_n_packing_is_exact_without_cross_task_products() -> None:
    """Independent rows occupy free N columns; physical contraction stays K=256."""
    rng = np.random.default_rng(20260903)
    rows = 17
    x = rng.standard_normal((rows, 256), dtype=np.float32)
    weight = rng.standard_normal((256, 1024), dtype=np.float32)
    conventional = x @ weight
    n_packed = weight.T @ x.T
    assert n_packed.shape == (1024, rows)
    np.testing.assert_allclose(n_packed.T, conventional, rtol=2e-5, atol=2e-5)


def coded_mixing_matrix(width: int) -> np.ndarray:
    """Well-conditioned signed transforms satisfying SM120 2:4 groups."""
    h2 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    if width == 4:
        # Two independent H2 blocks: exactly two nonzeros in each row.
        return np.kron(np.eye(2, dtype=np.float32), h2)
    if width == 8:
        # H4 kron I2 has two nonzeros in columns 0:4 and two in 4:8.
        h4 = np.kron(h2, h2)
        return np.kron(h4, np.eye(2, dtype=np.float32))
    raise ValueError(width)


def coded_pack_activations(xs: np.ndarray) -> np.ndarray:
    """Pack MxRxK activations in physical [k, task] order."""
    m, width, k = xs.shape
    return xs.transpose(0, 2, 1).reshape(m, k * width)


def coded_pack_weights(weight: np.ndarray, mixing: np.ndarray) -> np.ndarray:
    """Create physical (R*K)x(R*N) coded sparse operand."""
    k, n = weight.shape
    width = mixing.shape[0]
    packed = np.empty((k, width, width, n), dtype=weight.dtype)
    for out_task in range(width):
        for in_task in range(width):
            packed[:, in_task, out_task, :] = (
                mixing[out_task, in_task] * weight
            )
    return packed.reshape(k * width, width * n)


def _check_coded_width(width: int) -> None:
    rng = np.random.default_rng(20260902 + width)
    xs = rng.standard_normal((7, width, 256), dtype=np.float32)
    weight = rng.standard_normal((256, 31), dtype=np.float32)
    mixing = coded_mixing_matrix(width)
    packed_x = coded_pack_activations(xs)
    packed_w = coded_pack_weights(weight, mixing)
    coded_y = (packed_x @ packed_w).reshape(7, width, 31)
    decoded_y = np.einsum("ij,mjn->min", np.linalg.inv(mixing), coded_y)
    expected_y = np.einsum("mik,kn->min", xs, weight)
    np.testing.assert_allclose(decoded_y, expected_y, rtol=3e-5, atol=4e-5)

    # Every hardware group of four K values contains exactly two structural
    # nonzeros for every output channel.  For width=8 this is checked in both
    # four-task groups at every original k.
    structural = coded_pack_weights(
        np.ones_like(weight), mixing).reshape(-1, 4, width * weight.shape[1])
    assert np.all(np.count_nonzero(structural, axis=1) == 2)
    np.testing.assert_allclose(
        mixing @ mixing.T,
        np.eye(width, dtype=np.float32) * np.count_nonzero(mixing[0]),
    )


def test_four_way_coded_ff1_is_exact_and_two_of_four() -> None:
    _check_coded_width(4)


def test_eight_way_coded_ff1_is_exact_and_two_of_four() -> None:
    _check_coded_width(8)


def test_coded_useful_flop_accounting() -> None:
    original_m = 32768
    for width in (2, 4, 8):
        useful = width * (2 * original_m * 256 * 1024)
        physical_dense = 2 * (width * 1024) * original_m * (width * 256)
        assert physical_dense == width * useful


def cyclic_latin_square(width: int) -> np.ndarray:
    """The proposed task-id placement, zero based.

    For width four this is exactly:
      1 4 3 2
      2 1 4 3
      3 2 1 4
      4 3 2 1
    after adding one to the returned identifiers.
    """
    row = np.arange(width, dtype=np.int32)[:, None]
    column = np.arange(width, dtype=np.int32)[None, :]
    return (row - column) % width


def test_cyclic_latin_layout_matches_proposed_four_way_placement() -> None:
    expected = np.array(
        [[1, 4, 3, 2],
         [2, 1, 4, 3],
         [3, 2, 1, 4],
         [4, 3, 2, 1]],
        dtype=np.int32,
    )
    actual = cyclic_latin_square(4) + 1
    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.sort(actual, axis=0) == np.arange(1, 5)[:, None])
    assert np.all(np.sort(actual, axis=1) == np.arange(1, 5)[None, :])


def test_latin_two_of_four_phases_cover_every_cell_once() -> None:
    """Two complementary sparse phases cover the proposed layout exactly."""
    for width in (4, 8):
        latin = cyclic_latin_square(width)
        covered = np.zeros_like(latin, dtype=np.int32)
        for phase_positions in ((0, 2), (1, 3)):
            for quartet_start in range(0, width, 4):
                for offset in phase_positions:
                    covered[:, quartet_start + offset] += 1
        assert np.all(covered == 1)


def test_one_sparse_mma_cannot_route_latin_group_to_independent_outputs() -> None:
    """Expose the contraction boundary before writing a CUDA implementation.

    Metadata can select two K positions, but both selected products update the
    same C[m,n].  The proposed row therefore contains two different task IDs
    in every phase.  A permutation can move that scalar C element, but cannot
    split its already-summed contributions into two task-local outputs.
    """
    for width in (4, 8):
        latin = cyclic_latin_square(width)
        for row in range(width):
            for quartet_start in range(0, width, 4):
                for phase_positions in ((0, 2), (1, 3)):
                    selected = {
                        int(latin[row, quartet_start + offset])
                        for offset in phase_positions
                    }
                    assert len(selected) == 2


def test_latin_single_accumulator_scalar_counterexample() -> None:
    """Numerically show the cross-task sum produced by one C accumulator."""
    latin = cyclic_latin_square(4)
    task_products = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    phase_02 = np.array([
        task_products[latin[row, 0]] + task_products[latin[row, 2]]
        for row in range(4)
    ])
    phase_13 = np.array([
        task_products[latin[row, 1]] + task_products[latin[row, 3]]
        for row in range(4)
    ])
    np.testing.assert_array_equal(phase_02, np.array([7, 10, 7, 10]))
    np.testing.assert_array_equal(phase_13, np.array([10, 7, 10, 7]))
    # Only two distinct pair sums survive, not four independent task products.
    assert len(np.unique(np.concatenate((phase_02, phase_13)))) == 2
