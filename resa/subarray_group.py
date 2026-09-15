"""The only module allowed to touch scalesim.compute.* (CLAUDE.md rule 1).

It does one thing: hand SCALE-Sim a config that reports the fissioned logical
array dimensions, run the stock compute model, and return the demand matrices.
Keeping it thin is the point -- when SCALE-Sim is upgraded, this is the single
file that has to be re-checked.

Demand matrix semantics (verified in Phase 0, docs/resa_phase0_notes.md):
one row = one cycle, one column = one row/col port of the array, value =
operand address or -1 for a null request. Matrices come back from SCALE-Sim as
float64 but only ever hold addresses and -1, so they are narrowed to int32 on
receipt, which halves memory.
"""

import copy

import numpy as np

from scalesim.compute.operand_matrix import operand_matrix
from scalesim.compute.systolic_compute_is import systolic_compute_is
from scalesim.compute.systolic_compute_os import systolic_compute_os
from scalesim.compute.systolic_compute_ws import systolic_compute_ws
from scalesim.layout_utils import layouts

# Demand matrices hold addresses and -1. int32 is exact for every address the
# project can generate, which assert_addresses_fit_int32() enforces at runtime
# rather than only in the tests -- the bound moves if anyone edits the
# FilterOffset / OfmapOffset in a config.
DEMAND_DTYPE = np.int32
_MAX_ADDRESS = int(np.iinfo(DEMAND_DTYPE).max)

_COMPUTE_CLASSES = {
    'ws': systolic_compute_ws,
    'is': systolic_compute_is,
    'os': systolic_compute_os,
}


def assert_addresses_fit_int32(*matrices):
    """Guard the int32 narrowing. Raises before any silent truncation happens."""
    for matrix in matrices:
        if matrix.size == 0:
            continue
        highest = int(np.max(matrix))
        assert highest <= _MAX_ADDRESS, (
            'operand address %d exceeds int32; demand matrices cannot be '
            'narrowed. Check the *Offset values in the config.' % highest)
        lowest = int(np.min(matrix))
        assert lowest >= -1, 'unexpected operand value %d (only -1 is null)' % lowest


def to_demand_dtype(matrix):
    """Narrow a demand matrix to int32, refusing to do it if that would lose data."""
    narrowed = matrix.astype(DEMAND_DTYPE)
    assert np.array_equal(narrowed, matrix), \
        'demand matrix is not integral; int32 narrowing would lose data'
    return narrowed


def build_operand_matrices(scale_cfg, topo_obj, layer_id, layout_obj=None):
    """Whole-layer IFMAP / filter / OFMAP operand matrices for one layer.

    Tiling happens by slicing these (resa/tensor_split.py); the conv-to-GEMM
    lowering is never reimplemented.
    """
    op_mat = operand_matrix()
    op_mat.set_params(config_obj=scale_cfg,
                      topoutil_obj=topo_obj,
                      layoututil_obj=layout_obj if layout_obj is not None else layouts(),
                      layer_id=layer_id)
    _, ifmap_op_mat = op_mat.get_ifmap_matrix()
    _, filter_op_mat = op_mat.get_filter_matrix()
    _, ofmap_op_mat = op_mat.get_ofmap_matrix()

    assert_addresses_fit_int32(ifmap_op_mat, filter_op_mat, ofmap_op_mat)
    return ifmap_op_mat, filter_op_mat, ofmap_op_mat


class subarray_group:
    """One logical array (a fission of the sub-array grid) running one tile."""

    def __init__(self, scale_cfg, logical_rows, logical_cols, dataflow='ws'):
        assert dataflow in _COMPUTE_CLASSES, 'unknown dataflow: %s' % dataflow
        assert logical_rows > 0 and logical_cols > 0
        # Sparsity is out of scope (CLAUDE.md rule 7) and would change the
        # operand matrices underneath us.
        assert not scale_cfg.sparsity_support, \
            'sparsity is out of scope for this project'

        # A private copy: callers inject different dims per fission and must not
        # disturb each other's config.
        self.config = copy.deepcopy(scale_cfg)
        self.config.set_arr_dims(rows=logical_rows, cols=logical_cols)
        self.config.set_dataflow(dataflow)

        self.logical_rows = logical_rows
        self.logical_cols = logical_cols
        self.dataflow = dataflow
        self.compute_system = None

    #
    @classmethod
    def from_fission(cls, scale_cfg, fission_obj, dataflow='ws'):
        """Build the group for a fission, taking the logical dims from it."""
        arr_row, arr_col = fission_obj.logical_dims
        return cls(scale_cfg, arr_row, arr_col, dataflow=dataflow)

    #
    def run(self, ifmap_op_mat, filter_op_mat, ofmap_op_mat):
        """Return (ifmap, filter, ofmap) demand matrices as int32.

        Mirrors what single_layer_sim.run() does for the compute half: WS takes
        four extra sparsity arguments that IS and OS do not
        (single_layer_sim.py:207-220). With sparsity off they are unused, but
        the branch is kept so the call sites stay recognisable.
        """
        assert_addresses_fit_int32(ifmap_op_mat, filter_op_mat, ofmap_op_mat)

        compute_system = _COMPUTE_CLASSES[self.dataflow]()
        if self.dataflow == 'ws':
            compute_system.set_params(config_obj=self.config,
                                      ifmap_op_mat=ifmap_op_mat,
                                      filter_op_mat=filter_op_mat,
                                      ofmap_op_mat=ofmap_op_mat,
                                      sparsity_ratio_N=1,
                                      sparsity_ratio_M=1)
        else:
            compute_system.set_params(config_obj=self.config,
                                      ifmap_op_mat=ifmap_op_mat,
                                      filter_op_mat=filter_op_mat,
                                      ofmap_op_mat=ofmap_op_mat)

        compute_system.create_demand_matrices()
        self.compute_system = compute_system

        return tuple(to_demand_dtype(matrix)
                     for matrix in compute_system.get_demand_matrices())

    #
    def get_access_counts(self):
        """(ifmap reads, filter reads, ofmap writes) counted by SCALE-Sim.

        These are the numbers the buffer-port contention model consumes in
        Phase 3, and the ones the tiling invariants are written against.
        """
        assert self.compute_system is not None, 'run() first'
        return (self.compute_system.get_ifmap_requests(),
                self.compute_system.get_filter_requests(),
                self.compute_system.get_ofmap_requests())

    def get_utilisation(self):
        """(average mapping efficiency, average compute utilisation), fractions."""
        assert self.compute_system is not None, 'run() first'
        return (self.compute_system.get_avg_mapping_efficiency(),
                self.compute_system.get_avg_compute_utilization())

    def get_fold_counts(self):
        """(row folds, col folds) SCALE-Sim used for the demand matrices."""
        assert self.compute_system is not None, 'run() first'
        return (self.compute_system.row_fold_demand_matrices,
                self.compute_system.col_fold)

    def __repr__(self):
        return ('subarray_group(%dx%d, %s)'
                % (self.logical_rows, self.logical_cols, self.dataflow))
