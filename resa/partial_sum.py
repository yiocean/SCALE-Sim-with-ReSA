"""The read half of a K-fold accumulation, which SCALE-Sim does not emit.

cycle_model.partial_sum_reads() gives the count; this gives the schedule, which
is what the cycle-accurate arbiter needs. The construction is exact rather than
synthetic: every ofmap write that is not part of row fold 0 is preceded by a
read of the same address, so the read-back stream IS the ofmap demand matrix
with row fold 0 blanked out. Same addresses, same cycles, same ports.

Fold layout (read off the compute classes, not assumed):

    systolic_compute_ws.py:261, _is.py:227, _os.py:271 all emit

        for fc in range(col_fold):
            for fr in range(row_fold):

    so the rf * cf folds of a demand matrix are col-fold outer, row-fold inner,
    each occupying the same number of rows. Fold index f therefore has row fold
    index f % rf, and the folds needing no read-back are exactly f % rf == 0.

OS gets no read-back matrix at all: it folds over M and N and accumulates K in
place in the PE (cycle_model.partial_sum_reads).
"""

import numpy as np

from resa.subarray_group import DEMAND_DTYPE

NULL = -1


def read_matrix(ofmap_demand, dataflow, row_folds, col_folds):
    """Partial-sum read-back demand matrix, or None when there is none.

    Returns a matrix of the same shape as `ofmap_demand` in which the folds of
    row fold 0 are all NULL. Rides the same ports as the ofmap write, so the
    caller maps it with the ofmap entry of port_map.
    """
    assert row_folds >= 1 and col_folds >= 1, 'fold counts must be positive'

    if dataflow == 'os' or row_folds == 1:
        return None

    total_rows = ofmap_demand.shape[0]
    num_folds = row_folds * col_folds
    assert total_rows % num_folds == 0, (
        'ofmap demand matrix has %d rows, not divisible by %d folds; the fold '
        'layout assumption in resa/partial_sum.py no longer holds'
        % (total_rows, num_folds))
    fold_rows = total_rows // num_folds

    reads = ofmap_demand.astype(DEMAND_DTYPE, copy=True)
    for fold in range(0, num_folds, row_folds):        # f % row_folds == 0
        reads[fold * fold_rows:(fold + 1) * fold_rows] = NULL
    return reads


def count(ofmap_demand, dataflow, row_folds, col_folds):
    """Non-null entries of read_matrix(), without materialising it."""
    reads = read_matrix(ofmap_demand, dataflow, row_folds, col_folds)
    if reads is None:
        return 0
    return int(np.count_nonzero(reads != NULL))


def live_words(dataflow, m, n, k, arr_row, arr_col):
    """Partial sums a tile must hold at once, in SCALE-Sim words.

    NOT M * N. The fold loop is col-fold OUTER, row-fold inner (see the module
    docstring), so a tile finishes every K slice of one column block before it
    moves to the next. Only that block is mid-accumulation:

        WS   col folds run over N  ->  M * min(N, arr_col)
        IS   col folds run over M  ->  min(M, arr_col) * N
        OS   nothing: K accumulates in the PE

    This is what decides whether the read-modify-write stays on chip. If the
    live set exceeds the group's ofmap allocation the partial sums spill to
    DRAM, and the off-chip traffic in resa/memory_policy.py (which counts the
    final output once) stops being an upper bound on reuse.
    """
    from resa import cycle_model

    if dataflow == 'os':
        return 0
    row_folds, _ = cycle_model.fold_counts(dataflow, m, n, k, arr_row, arr_col)
    if row_folds == 1:
        return 0
    if dataflow == 'ws':
        return m * min(n, arr_col)
    return min(m, arr_col) * n


def fits_on_chip(resa_cfg, dataflow, m, n, k, arr_row, arr_col):
    """(fits, live words, budget words) for one tile's partial-sum working set."""
    live = live_words(dataflow, m, n, k, arr_row, arr_col)
    budget = resa_cfg.get_operand_buffer_words_per_group()
    return live <= budget, live, budget
