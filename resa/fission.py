"""Sub-array fission: the nine legal ways to fuse the 4x4 sub-array grid.

docs/resa_spec.md section 2. A fission (Fr, Fc) fuses Fr x Fc adjacent
sub-arrays into one logical array; the grid is partitioned into
16 / (Fr * Fc) such logical arrays, each running an independent thread.

The hard constraint from the paper: at any instant every sub-array uses the
same fission. Half the chip cannot be 2x2 while the other half is 1x1. That is
an assertion here, not a recommendation -- see assert_uniform().

There is no ReSA logic in generating a demand matrix for a fissioned array:
resa/subarray_group.py just hands SCALE-Sim a config reporting the logical
dimensions. What this module owns is which dimensions are legal.
"""

import math


class fission:
    """One legal (Fr, Fc) combination and the logical array it produces."""

    __slots__ = ('rows', 'cols', 'sub_array_rows', 'sub_array_cols',
                 'grid_rows', 'grid_cols')

    def __init__(self, rows, cols, sub_array_rows, sub_array_cols,
                 grid_rows, grid_cols):
        assert grid_rows % rows == 0, \
            'fission rows must tile the grid: %d does not divide %d' % (rows, grid_rows)
        assert grid_cols % cols == 0, \
            'fission cols must tile the grid: %d does not divide %d' % (cols, grid_cols)

        self.rows = rows
        self.cols = cols
        self.sub_array_rows = sub_array_rows
        self.sub_array_cols = sub_array_cols
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

    #
    @property
    def logical_dims(self):
        """(arr_row, arr_col) of the fused array, the dims handed to SCALE-Sim."""
        return self.rows * self.sub_array_rows, self.cols * self.sub_array_cols

    @property
    def sub_arrays_per_thread(self):
        return self.rows * self.cols

    @property
    def num_threads(self):
        """Independent logical arrays available at this fission."""
        return (self.grid_rows * self.grid_cols) // self.sub_arrays_per_thread

    @property
    def name(self):
        return '%dx%d' % (self.rows, self.cols)

    #
    def thread_origin(self, thread_id):
        """Top-left sub-array (grid row, grid col) of one thread.

        Threads are numbered in row-major order over the grid of logical
        arrays. The NoC is unidirectional (left-to-right, top-to-bottom), so
        the origin is also the sub-array that operands enter through.
        """
        assert 0 <= thread_id < self.num_threads, 'thread id out of range'
        threads_per_row = self.grid_cols // self.cols
        return ((thread_id // threads_per_row) * self.rows,
                (thread_id % threads_per_row) * self.cols)

    def sub_array_ids(self, thread_id):
        """Flat ids (row-major over the grid) of the sub-arrays of one thread."""
        origin_row, origin_col = self.thread_origin(thread_id)
        return [ (origin_row + r) * self.grid_cols + (origin_col + c)
                 for r in range(self.rows) for c in range(self.cols) ]

    #
    def predicted_demand_rows(self, num_rows_operand, window_size, num_filters,
                              dataflow='ws'):
        """Rows of the demand matrix this fission produces for one tile.

        Delegates to resa/cycle_model.py, which carries the law for each
        dataflow. Arguments are (M, K, N) in GEMM terms.
        """
        from resa import cycle_model
        arr_row, arr_col = self.logical_dims
        return cycle_model.demand_rows(dataflow, num_rows_operand, num_filters,
                                       window_size, arr_row, arr_col)

    #
    def __eq__(self, other):
        return (isinstance(other, fission)
                and (self.rows, self.cols) == (other.rows, other.cols)
                and self.logical_dims == other.logical_dims)

    def __hash__(self):
        return hash((self.rows, self.cols, self.sub_array_rows, self.sub_array_cols))

    def __repr__(self):
        arr_row, arr_col = self.logical_dims
        return ('fission(%dx%d -> %dx%d array, %d threads)'
                % (self.rows, self.cols, arr_row, arr_col, self.num_threads))


class fission_space:
    """Every legal fission for a given sub-array grid.

    For the paper's 4x4 grid this is the nine combinations of
    Fr in {1, 2, 4} and Fc in {1, 2, 4}, matching spec section 2's table.
    """

    def __init__(self, resa_cfg):
        self.sub_array_rows, self.sub_array_cols = resa_cfg.get_sub_array_dims()
        self.grid_rows, self.grid_cols = resa_cfg.get_sub_array_grid()

        self._all = [
            fission(rows, cols, self.sub_array_rows, self.sub_array_cols,
                    self.grid_rows, self.grid_cols)
            for rows in _divisors(self.grid_rows)
            for cols in _divisors(self.grid_cols)
        ]

    #
    def all(self):
        """Legal fissions, smallest logical array first."""
        return list(self._all)

    def get(self, rows, cols):
        for candidate in self._all:
            if candidate.rows == rows and candidate.cols == cols:
                return candidate
        raise KeyError('illegal fission %dx%d for a %dx%d grid'
                       % (rows, cols, self.grid_rows, self.grid_cols))

    def monolithic(self):
        """The fission that fuses the whole chip into one array (4x4 here).

        This is the regression baseline: one 128x128 weight-stationary array,
        which is exactly spec section 9's baseline.
        """
        return self.get(self.grid_rows, self.grid_cols)

    def __len__(self):
        return len(self._all)


def assert_uniform(fissions):
    """Every sub-array must use the same fission at any instant (spec section 2).

    The paper is explicit: "all sub-arrays should connect with the same
    combination of the sub-array". Mixed fissions would be a different machine.
    """
    fission_list = list(fissions)
    assert fission_list, 'no fission given'
    first = fission_list[0]
    for other in fission_list[1:]:
        assert other == first, (
            'all sub-arrays must use the same fission at any instant '
            '(spec section 2): found %s alongside %s' % (other.name, first.name))
    return first


def _divisors(value):
    return [d for d in range(1, value + 1) if value % d == 0]
