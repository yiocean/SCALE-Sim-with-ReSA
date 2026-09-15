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
