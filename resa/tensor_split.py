"""Tensor splitting: cut a layer into tiles along M and N, never along K.

docs/resa_plan.md, tensor_split.py. The conv-to-GEMM lowering is never
reimplemented: resa/subarray_group.py builds the whole-layer operand matrices
with SCALE-Sim's own operand_matrix, and this module slices them.

    ReSA parameter  GEMM dim  operand matrix                        split?
    Ow x Oh         M         ifmap rows (T), ofmap rows             yes
    F               N         filter cols (Sc), ofmap cols           yes
    C x kw x kh     K         ifmap cols / filter rows (Sr)          disabled

K is disabled on purpose (docs/resa_spec.md section 8, item 1): splitting it
would need cross-sub-array partial-sum reduction, which SCALE-Sim does not
model. Every tile therefore sees the full conv window, and tiles are
independent -- no tile ever needs another tile's partial sums.

Access counts as a function of the tiling have closed forms, which is what
tests/test_tensor_split.py checks; see predicted_access_counts().
"""

import math


class tile:
    """One independent piece of work: rows [m_start, m_start+m_len) x cols."""

    __slots__ = ('tile_id', 'm_start', 'm_len', 'n_start', 'n_len')

    def __init__(self, tile_id, m_start, m_len, n_start, n_len):
        assert m_len > 0 and n_len > 0, 'empty tile'
        assert m_start >= 0 and n_start >= 0
        self.tile_id = tile_id
        self.m_start = m_start
        self.m_len = m_len
        self.n_start = n_start
        self.n_len = n_len

    @property
    def m_stop(self):
        return self.m_start + self.m_len

    @property
    def n_stop(self):
        return self.n_start + self.n_len

    def __repr__(self):
        return ('tile(%d: M[%d:%d] N[%d:%d])'
                % (self.tile_id, self.m_start, self.m_stop,
                   self.n_start, self.n_stop))


class layer_tiling:
    """A uniform tiling of one layer's M and N dimensions.

    Tiles are emitted M-outer, N-inner, matching Algorithm 1's loop nest
    (spec section 4). The last tile along each dimension may be short.
    """

    def __init__(self, num_operand_rows, num_filters,
                 m_tile_size=None, n_tile_size=None):
        assert num_operand_rows > 0 and num_filters > 0

        self.num_operand_rows = int(num_operand_rows)   # T
        self.num_filters = int(num_filters)             # Sc

        self.m_tile_size = self.num_operand_rows if m_tile_size is None \
            else int(m_tile_size)
        self.n_tile_size = self.num_filters if n_tile_size is None \
            else int(n_tile_size)

        assert 1 <= self.m_tile_size <= self.num_operand_rows, \
            'M tile size %d out of range for T = %d' % (self.m_tile_size,
                                                        self.num_operand_rows)
        assert 1 <= self.n_tile_size <= self.num_filters, \
            'N tile size %d out of range for F = %d' % (self.n_tile_size,
                                                        self.num_filters)

        self.num_m_tiles = math.ceil(self.num_operand_rows / self.m_tile_size)
        self.num_n_tiles = math.ceil(self.num_filters / self.n_tile_size)

    #
    @classmethod
    def from_ofmap_tile(cls, ofmap_rows, ofmap_cols, oh_tile, ow_tile,
                        num_filters, n_tile_size=None):
        """Build a tiling from Algorithm 1's (oh, ow) tile sizes.

        The operand matrix numbers its rows in ofmap raster order, i.e. row
        index = ofmap_row * Ow + ofmap_col. An (oh x ow) tile is therefore a
        contiguous range of M **only when ow_tile spans the full ofmap width**;
        a narrower tile picks strided groups of rows and cannot be expressed as
        a slice. Phase 2 owns that case -- it is refused loudly here rather
        than silently mis-sliced.
        """
        assert 1 <= oh_tile <= ofmap_rows
        assert 1 <= ow_tile <= ofmap_cols
        if ow_tile != ofmap_cols:
            raise NotImplementedError(
                'ow tile %d < ofmap width %d selects non-contiguous operand '
                'matrix rows; strided tiles are Phase 2 work'
                % (ow_tile, ofmap_cols))
        return cls(ofmap_rows * ofmap_cols, num_filters,
                   m_tile_size=oh_tile * ofmap_cols, n_tile_size=n_tile_size)

    #
    def tiles(self):
        """Every tile, M-outer and N-inner."""
        out = []
        tile_id = 0
        for m_index in range(self.num_m_tiles):
            m_start = m_index * self.m_tile_size
            m_len = min(self.m_tile_size, self.num_operand_rows - m_start)
            for n_index in range(self.num_n_tiles):
                n_start = n_index * self.n_tile_size
                n_len = min(self.n_tile_size, self.num_filters - n_start)
                out.append(tile(tile_id, m_start, m_len, n_start, n_len))
                tile_id += 1
        return out

    @property
    def num_tiles(self):
        return self.num_m_tiles * self.num_n_tiles

    @property
    def is_whole_layer(self):
        return self.num_tiles == 1

    #
    def predicted_access_counts(self, window_size, arr_row, arr_col):
        """Closed forms for the SRAM accesses this tiling causes.

        From the per-tile fold law verified in Phase 0, summed over tiles:

            filter reads = Sr * Sc * num_M_tiles
                weights are reloaded once per M tile
            ifmap reads  = Sr * T * sum_n ceil(n_len / arr_col)
            ofmap writes = T * Sc * ceil(Sr / arr_row)
                independent of the tiling
            MACs         = T * Sc * Sr
                independent of the tiling

        Returns (ifmap reads, filter reads, ofmap writes, MACs).
        """
        assert window_size > 0 and arr_row > 0 and arr_col > 0

        col_folds_total = sum(math.ceil(t.n_len / arr_col)
                              for t in self.tiles()[:self.num_n_tiles])
        row_folds = math.ceil(window_size / arr_row)

        ifmap_reads = window_size * self.num_operand_rows * col_folds_total
        filter_reads = window_size * self.num_filters * self.num_m_tiles
        ofmap_writes = self.num_operand_rows * self.num_filters * row_folds
        macs = self.num_operand_rows * self.num_filters * window_size

        return ifmap_reads, filter_reads, ofmap_writes, macs

    def __repr__(self):
        return ('layer_tiling(T=%d F=%d, %d x %d tiles of %d x %d)'
                % (self.num_operand_rows, self.num_filters,
                   self.num_m_tiles, self.num_n_tiles,
                   self.m_tile_size, self.n_tile_size))


def split_operands(ifmap_op_mat, filter_op_mat, ofmap_op_mat, tile_obj):
    """Slice the whole-layer operand matrices down to one tile.

    Returns views, not copies. K is never sliced, so the conv window survives
    whole in every tile and the tiles stay independent.
    """
    assert ifmap_op_mat.shape[1] == filter_op_mat.shape[0], \
        'operand matrices disagree on K'
    assert tile_obj.m_stop <= ifmap_op_mat.shape[0], 'M tile runs past the layer'
    assert tile_obj.n_stop <= filter_op_mat.shape[1], 'N tile runs past the layer'

    ifmap_tile = ifmap_op_mat[tile_obj.m_start:tile_obj.m_stop, :]
    filter_tile = filter_op_mat[:, tile_obj.n_start:tile_obj.n_stop]
    ofmap_tile = ofmap_op_mat[tile_obj.m_start:tile_obj.m_stop,
                              tile_obj.n_start:tile_obj.n_stop]

    assert ifmap_tile.shape[1] == filter_tile.shape[0] == ifmap_op_mat.shape[1], \
        'K was split; that is disabled (spec section 8, item 1)'
    return ifmap_tile, filter_tile, ofmap_tile
