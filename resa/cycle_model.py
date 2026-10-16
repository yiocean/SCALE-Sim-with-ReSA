"""Closed-form demand matrix size laws, one per dataflow.

Algorithm 1's search space is millions of points (UNet_maestro's Conv1_2 alone
is ~23k: 11 ow x 11 oh x 7 oc x 3 dataflows x 9 fissions). Materialising a
demand matrix per point is not an option, so Examine() is pure arithmetic on
the laws below.

These are NOT the paper's analytical model. docs/resa_spec.md section 6 is
banned from producing numbers (CLAUDE.md rule 6) because it is unvalidated and
has a known directional bias: it omits padding cost, and padding waste is the
whole point of ReSA, so it underestimates exactly on the low-utilisation layers
that matter. What is below was derived from SCALE-Sim's own compute classes and
checked against their output bit-exactly, across every dataflow, nine fissions
and a spread of layer shapes (tests/test_cycle_model.py). Every layer's chosen
configuration is then re-verified against a real run in tile_search.

GEMM naming used throughout, independent of what each compute class calls its
own attributes (they disagree -- WS's self.T is M, OS's self.T is K):

    M = Ow * Oh   ofmap pixels        rows of the ifmap operand matrix
    N = F         filters             cols of the filter operand matrix
    K = kw*kh*C   conv window         cols of ifmap / rows of filter

R x C is the logical array, i.e. (32*Fr) x (32*Fc) for a fission (Fr, Fc).
"""

import math

DATAFLOWS = ('ws', 'is', 'os')


def fold_counts(dataflow, m, n, k, arr_row, arr_col):
    """(row folds, col folds) for one tile.

    Which GEMM dimension is folded onto which array axis is what actually
    differs between the dataflows:

        WS  weights are held: K on rows, N on cols, M streams in time
        IS  inputs are held:  K on rows, M on cols, N streams in time
        OS  outputs are held: M on rows, N on cols, K streams in time
    """
    _check(dataflow, m, n, k, arr_row, arr_col)

    if dataflow == 'ws':
        return math.ceil(k / arr_row), math.ceil(n / arr_col)
    if dataflow == 'is':
        return math.ceil(k / arr_row), math.ceil(m / arr_col)
    return math.ceil(m / arr_row), math.ceil(n / arr_col)


def rows_per_fold(dataflow, m, n, k, arr_row, arr_col):
    """Demand matrix rows one fold occupies.

    WS and IS both pay arr_row cycles to load the stationary operand, the
    streaming dimension, arr_row - 1 of skew and arr_col - 1 of drain. OS holds
    no operand up front, so it only pays the streaming dimension plus the two
    pipeline terms.
    """
    _check(dataflow, m, n, k, arr_row, arr_col)

    if dataflow == 'ws':
        streamed = m
    elif dataflow == 'is':
        streamed = n
    else:
        return k + arr_row + arr_col - 2

    return 2 * arr_row + arr_col + streamed - 2


def demand_rows(dataflow, m, n, k, arr_row, arr_col):
    """Total demand matrix rows, i.e. compute cycles, for one tile."""
    row_folds, col_folds = fold_counts(dataflow, m, n, k, arr_row, arr_col)
    return row_folds * col_folds * rows_per_fold(dataflow, m, n, k,
                                                 arr_row, arr_col)


def demand_rows_for_fission(dataflow, m, n, k, fission_obj):
    """demand_rows() with the array dimensions taken from a fission."""
    arr_row, arr_col = fission_obj.logical_dims
    return demand_rows(dataflow, m, n, k, arr_row, arr_col)


def access_counts(dataflow, m, n, k, arr_row, arr_col, valid_ifmap=None):
    """Non-null entries of each demand matrix, i.e. SRAM accesses for one tile.

    Returns (ifmap reads, filter reads, ofmap writes). Verified against real
    demand matrices for all three dataflows in tests/test_access_model.py.

        WS   ifmap = V * col_folds,  filter = K * N,             ofmap = M * N * row_folds
        IS   ifmap = V,              filter = K * N * col_folds, ofmap = M * N * row_folds
        OS   ifmap = V * col_folds,  filter = K * N * row_folds, ofmap = M * N

    V is the number of *valid* ifmap operand entries, which is M * K only when
    every convolution window lies inside the input. It does not when the
    topology's ofmap dimensions round up past what the input supports -- with
    stride 2, ResNet50's CB3a_1 has 14,592 of its 215,296 window positions
    outside the 56x56 input, and operand_matrix fills those with -1. Passing
    valid_ifmap keeps the count honest; omitting it assumes M * K.

    Note this is what the arbiter consumes: it counts entries of the demand
    matrices, not SCALE-Sim's own ifmap_reads / ofmap_writes counters. Those
    differ for OS, where systolic_compute_os adds a per-fold perimeter term to
    ofmap_writes with + where the surrounding code uses *.
    """
    _check(dataflow, m, n, k, arr_row, arr_col)
    if valid_ifmap is None:
        valid_ifmap = m * k
    assert 0 <= valid_ifmap <= m * k, 'valid ifmap entries out of range'

    row_folds, col_folds = fold_counts(dataflow, m, n, k, arr_row, arr_col)

    if dataflow == 'ws':
        return valid_ifmap * col_folds, k * n, m * n * row_folds
    if dataflow == 'is':
        return valid_ifmap, k * n * col_folds, m * n * row_folds
    return valid_ifmap * col_folds, k * n * row_folds, m * n


def partial_sum_reads(dataflow, m, n, k, arr_row, arr_col):
    """Partial-sum read-backs one tile needs, on top of access_counts().

        WS / IS   (row_folds - 1) * M * N
        OS        0

    SCALE-Sim models compute, not accumulation. When K exceeds the array height
    the reduction is folded over time: WS and IS put K on the rows, so a tile is
    swept row_folds times and each sweep produces a partial result over its own
    K slice. The ofmap demand matrix already carries the WRITE of every one of
    them -- that is why the ofmap access law is M * N * row_folds and not M * N.
    What SCALE-Sim never emits is the matching READ: fold i+1 has to fetch fold
    i's partial sum back to accumulate onto it. An accumulation is a
    read-modify-write and only the write was being charged.

    This is K folded in TIME, inside one thread, on one logical array, into that
    thread's own buffer group. It is not the cross-sub-array K split that
    CLAUDE.md rule 7 excludes -- no partial sum crosses a thread boundary and
    tensor_split.split_operands still refuses to slice K. The two are
    independent; this one has been in the model since Phase 0 by way of
    fold_counts(), only half-priced.

    OS is zero because it holds the output stationary and streams K through
    time, accumulating in place in the PE: its folds are over M and N, which
    need no reduction. That asymmetry is real and is why high-K layers should
    prefer OS once this is charged.

    Where the read-back lands is an estimate (docs/resa_spec.md section 8, item
    16): the paper gives each sub-array an accumulator on the drain edge that
    sends results back to the on-chip buffer (section 3.1), which fixes the PORT
    -- the same bottom-edge ofmap port as the write -- but says nothing about
    where a partial sum lives between two row folds. A per-sub-array accumulator
    is one column wide and cannot hold M * N, so it is assumed to go back to the
    buffer. Set [resa_buffer] PartialSumAccumulation = inline_accumulator to
    price it at zero and measure how much the conclusion depends on this.
    """
    _check(dataflow, m, n, k, arr_row, arr_col)
    if dataflow == 'os':
        return 0
    row_folds, _ = fold_counts(dataflow, m, n, k, arr_row, arr_col)
    return (row_folds - 1) * m * n


def mapping_efficiency(dataflow, m, n, k, arr_row, arr_col):
    """Average fraction of the PE grid holding a useful operand, over all folds.

    SCALE-Sim's own metric (single_layer_sim.py:338, "Mapping Efficiency %"),
    derived here instead of measured so the report can carry it without building
    a demand matrix. Each fold uses min(arr_row, what is left) rows and likewise
    columns, so the average factorises into the two axes:

        WS   rows fold over K, cols over N
        IS   rows fold over K, cols over M
        OS   rows fold over M, cols over N

    Dense only. Sparsity is out of scope and asserted off in subarray_group, so
    the WS path's sum_sparse term reduces to the same count.
    """
    _check(dataflow, m, n, k, arr_row, arr_col)
    rows, cols = _fold_axes(dataflow, m, n, k)
    row_folds, col_folds = fold_counts(dataflow, m, n, k, arr_row, arr_col)
    mean_rows = ((row_folds - 1) * arr_row + (rows - (row_folds - 1) * arr_row)) / row_folds
    mean_cols = ((col_folds - 1) * arr_col + (cols - (col_folds - 1) * arr_col)) / col_folds
    return (mean_rows / arr_row) * (mean_cols / arr_col)


def cycles_per_fold(dataflow, m, n, k, arr_row, arr_col):
    """Cycles SCALE-Sim charges one fold when computing compute utilisation.

    It is shape[0] + shape[1] - 1 of the pre-skew fold demand, which works out
    to rows_per_fold() for OS and rows_per_fold() + arr_col - 1 for WS and IS.
    Verified against the per-fold lists the compute classes build.
    """
    base = rows_per_fold(dataflow, m, n, k, arr_row, arr_col)
    return base if dataflow == 'os' else base + arr_col - 1


def compute_utilisation(dataflow, m, n, k, arr_row, arr_col):
    """Average fraction of PE-cycles doing useful work, over all folds.

    SCALE-Sim's "Compute Util %" (single_layer_sim.py:339), which differs from
    mapping efficiency by also charging the pipeline fill and drain: a fold maps
    mac_used PEs but occupies cycles_per_fold() cycles to push T operands
    through them.

        mac_used * T / (arr_row * arr_col * cycles_per_fold)

    T is the dimension streamed through the held one -- M for WS, N for IS, K
    for OS -- and cycles_per_fold is the same for every fold, so this reduces to
    T * mapping_efficiency / cycles_per_fold.
    """
    _check(dataflow, m, n, k, arr_row, arr_col)
    streamed = {'ws': m, 'is': n, 'os': k}[dataflow]
    cycles = cycles_per_fold(dataflow, m, n, k, arr_row, arr_col)
    return streamed * mapping_efficiency(dataflow, m, n, k, arr_row, arr_col) / cycles


def weighted_utilisation(dataflow, shapes, k, arr_row, arr_col):
    """(mapping efficiency, compute utilisation) over a layer's tile shapes.

    `shapes` is [(m_len, n_len, count)] as tile_search.tile_shapes produces. A
    layer is generally cut into tiles of two sizes -- full ones and a smaller
    remainder on each axis -- and the edge tiles map fewer PEs, so the layer
    figure is the tile figures weighted by how many tiles have that shape.

    Native SCALE-Sim has no equivalent weighting because it runs a layer as a
    single tile.
    """
    total = sum(count for _, _, count in shapes)
    if not total:
        return 0.0, 0.0
    mapping = sum(count * mapping_efficiency(dataflow, m_len, n_len, k,
                                             arr_row, arr_col)
                  for m_len, n_len, count in shapes) / total
    compute = sum(count * compute_utilisation(dataflow, m_len, n_len, k,
                                              arr_row, arr_col)
                  for m_len, n_len, count in shapes) / total
    return mapping, compute


def _fold_axes(dataflow, m, n, k):
    """(what folds onto rows, what folds onto cols) for a dataflow."""
    if dataflow == 'ws':
        return k, n
    if dataflow == 'is':
        return k, m
    return m, n


def valid_ifmap_rows(ifmap_op_mat):
    """Per-operand-row count of valid (non -1) ifmap entries.

    A prefix sum over this gives the V of any contiguous M tile without
    touching the matrix again, which is what the search needs to stay cheap.
    """
    import numpy as np
    return (ifmap_op_mat != -1).sum(axis=1).astype(np.int64)


def gemm_dims_from_operands(ifmap_op_mat, filter_op_mat):
    """(M, N, K) of whatever operand matrices were handed in."""
    return ifmap_op_mat.shape[0], filter_op_mat.shape[1], filter_op_mat.shape[0]


def gemm_dims_from_layer(topo_obj, layer_id):
    """(M, N, K) of a whole layer, without building any operand matrix."""
    ofmap_h, ofmap_w = topo_obj.get_layer_ofmap_dims(layer_id)
    return (int(ofmap_h * ofmap_w),
            int(topo_obj.get_layer_num_filters(layer_id)),
            int(topo_obj.get_layer_window_size(layer_id)))


def _check(dataflow, m, n, k, arr_row, arr_col):
    assert dataflow in DATAFLOWS, 'unknown dataflow: %s' % (dataflow,)
    assert m > 0 and n > 0 and k > 0, 'degenerate GEMM dims (%d, %d, %d)' % (m, n, k)
    assert arr_row > 0 and arr_col > 0, 'degenerate array %dx%d' % (arr_row, arr_col)
