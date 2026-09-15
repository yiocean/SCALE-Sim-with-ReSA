"""Algorithm 2: the three-state off-chip memory policy.

docs/resa_spec.md section 7. Off-chip bandwidth is a single global budget --
it is not split per buffer group -- so fission reaches DRAM through a
completely different mechanism than it reaches on-chip SRAM. There is no
arbitration here. What matters is tile size and reuse distance:

    smaller array -> smaller tiles -> more tiles -> weights reloaded more often
                                                 -> more halo re-fetch on ifmap

so the traffic itself changes with the configuration, and the policy decides
how much of it can hide behind computation.

    initial_read  = first input tile + first weight tile
    middle_read   = every other input and weight tile
    middle_write  = every output tile except the last
    final_write   = the last output tile

    memory_wait = initial + max(0, middle - computation) + final

Planaria's policy is the same accounting with everything pushed to the ends:
the whole layer's input and weights in the initial state, the whole output in
the final state, and nothing left in the middle to overlap. Both are computed
here from the same traffic figures, so the comparison is like for like. This is
Planaria's *memory policy*, not a Planaria architecture -- the plan defers that.
"""

import math

import numpy as np


class dram_traffic:
    """Off-chip words a layer moves under one tiling, in operand elements."""

    __slots__ = ('ifmap_words', 'filter_words', 'ofmap_words',
                 'first_ifmap_words', 'first_filter_words', 'last_ofmap_words',
                 'num_tiles', 'num_m_tiles', 'num_n_tiles')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0))

    @property
    def total_words(self):
        return self.ifmap_words + self.filter_words + self.ofmap_words

    @property
    def read_words(self):
        return self.ifmap_words + self.filter_words

    def __repr__(self):
        return ('dram_traffic(%d tiles: %d ifmap + %d filter + %d ofmap words)'
                % (self.num_tiles, self.ifmap_words, self.filter_words,
                   self.ofmap_words))


class ifmap_footprint:
    """Distinct ifmap words a contiguous output-pixel range touches.

    Counting with np.unique over the operand matrix is exact but hopeless at
    scale -- UNet_maestro's worst layer has 186 M entries, and Phase 4 wants
    this for several tilings per layer. So the touched pixels are marked on a
    small H x W mask instead: a contiguous raster range of output pixels covers
    at most three segments (a partial first output row, the full rows between,
    a partial last row), and each segment marks a strided sub-grid.

    Strided is the operative word. A first attempt treated each segment as a
    solid rectangle, which is only right when the stride is no larger than the
    filter: ResNet50's CB3a_1 has a 1x1 filter at stride 2, so it samples every
    other pixel and the rectangle over-counted by 3.9x. tests/test_memory_policy
    checks every count against np.unique, which is what caught it.

    Sizes repeat as the window slides, so results are cached on
    (m_start mod ofmap_w, m_len, whether the tile clips against the input).
    """

    def __init__(self, topo_obj, layer_id):
        ifmap_h, ifmap_w = topo_obj.get_layer_ifmap_dims(layer_id)
        filter_h, filter_w = topo_obj.get_layer_filter_dims(layer_id)
        stride_r, stride_c = topo_obj.get_layer_strides(layer_id)
        ofmap_h, ofmap_w = topo_obj.get_layer_ofmap_dims(layer_id)

        self.ifmap_h, self.ifmap_w = int(ifmap_h), int(ifmap_w)
        self.filter_h, self.filter_w = int(filter_h), int(filter_w)
        self.stride_r, self.stride_c = int(stride_r), int(stride_c)
        self.ofmap_h, self.ofmap_w = int(ofmap_h), int(ofmap_w)
        self.channels = int(topo_obj.get_layer_num_channels(layer_id))
        self._cache = {}
        self._mask = np.zeros((self.ifmap_h, self.ifmap_w), dtype=bool)

    def _segments(self, m_start, m_len):
        """(first output row, col range), full-row block, (last row, col range)."""
        first_row, first_col = divmod(m_start, self.ofmap_w)
        last_row, last_col = divmod(m_start + m_len - 1, self.ofmap_w)

        if first_row == last_row:
            return [(first_row, first_row, first_col, last_col + 1)]

        segments = [(first_row, first_row, first_col, self.ofmap_w)]
        if last_row - first_row > 1:
            segments.append((first_row + 1, last_row - 1, 0, self.ofmap_w))
        segments.append((last_row, last_row, 0, last_col + 1))
        return segments

    def _mark(self, row_lo, row_hi, col_lo, col_hi):
        """Mark the input pixels the output block [rows] x [cols) touches."""
        for row_offset in range(self.filter_h):
            row_start = row_lo * self.stride_r + row_offset
            if row_start >= self.ifmap_h:
                continue
            row_stop = min(row_hi * self.stride_r + row_offset + 1, self.ifmap_h)
            rows = slice(row_start, row_stop, self.stride_r)
            for col_offset in range(self.filter_w):
                col_start = col_lo * self.stride_c + col_offset
                if col_start >= self.ifmap_w:
                    continue
                col_stop = min((col_hi - 1) * self.stride_c + col_offset + 1,
                               self.ifmap_w)
                self._mask[rows, col_start:col_stop:self.stride_c] = True

    def words(self, m_start, m_len):
        """Distinct ifmap elements (pixels x channels) the range touches."""
        assert m_len > 0
        last_row = (m_start + m_len - 1) // self.ofmap_w
        clipped = last_row * self.stride_r + self.filter_h > self.ifmap_h
        key = (m_start % self.ofmap_w, m_len, clipped)
        if key in self._cache:
            return self._cache[key]

        self._mask[:] = False
        for row_lo, row_hi, col_lo, col_hi in self._segments(m_start, m_len):
            if col_lo < col_hi:
                self._mark(row_lo, row_hi, col_lo, col_hi)
        words = int(self._mask.sum()) * self.channels

        self._cache[key] = words
        return words


def layer_traffic(operands, tiling, footprint=None):
    """Distinct off-chip words a layer moves under `tiling`.

    Counted as DISTINCT addresses per tile, not operand matrix entries. That
    distinction is the whole point on the ifmap side: neighbouring convolution
    windows overlap, so a tile re-fetches a halo, and a finer tiling re-fetches
    more of it. Counting entries would hide exactly the effect being measured.

    Weights are re-read once per M tile and inputs once per N tile, which is
    the same reuse structure the SRAM access laws have at fold granularity.

    Pass `footprint` (an ifmap_footprint) to count analytically instead of with
    np.unique; the two agree exactly and only the analytic one scales.
    """
    ifmap_op_mat = operands.matrices[0]
    window_size = operands.k

    per_m_tile = []
    for m_index in range(tiling.num_m_tiles):
        m_start = m_index * tiling.m_tile_size
        m_len = min(tiling.m_tile_size, tiling.num_operand_rows - m_start)
        if footprint is not None:
            per_m_tile.append(footprint.words(m_start, m_len))
        else:
            block = ifmap_op_mat[m_start:m_start + m_len]
            per_m_tile.append(int(np.unique(block[block != -1]).size))

    n_lens = [min(tiling.n_tile_size, tiling.num_filters - index * tiling.n_tile_size)
              for index in range(tiling.num_n_tiles)]

    ifmap_words = sum(per_m_tile) * tiling.num_n_tiles
    filter_words = window_size * sum(n_lens) * tiling.num_m_tiles
    ofmap_words = tiling.num_operand_rows * tiling.num_filters

    return dram_traffic(
        ifmap_words=ifmap_words,
        filter_words=filter_words,
        ofmap_words=ofmap_words,
        first_ifmap_words=per_m_tile[0],
        first_filter_words=window_size * n_lens[0],
        last_ofmap_words=(tiling.num_operand_rows
                          - (tiling.num_m_tiles - 1) * tiling.m_tile_size) * n_lens[-1],
        num_tiles=tiling.num_tiles,
        num_m_tiles=tiling.num_m_tiles,
        num_n_tiles=tiling.num_n_tiles)


class memory_wait:
    """The three states and what they cost."""

    __slots__ = ('initial_words', 'middle_words', 'final_words',
                 'initial_cycles', 'middle_cycles', 'final_cycles',
                 'computation_cycles', 'wait_cycles', 'policy')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0))

    @property
    def hidden_cycles(self):
        """Middle-state traffic that computation absorbed."""
        return min(self.middle_cycles, self.computation_cycles)

    def __repr__(self):
        return ('memory_wait(%s: %d = %d initial + %d exposed middle + %d final)'
                % (self.policy, self.wait_cycles, self.initial_cycles,
                   max(0, self.middle_cycles - self.computation_cycles),
                   self.final_cycles))


def resa_wait(traffic, bandwidth_words_per_cycle, computation_cycles):
    """Algorithm 2: load one tile, start computing, hide the rest."""
    assert bandwidth_words_per_cycle > 0

    initial = traffic.first_ifmap_words + traffic.first_filter_words
    final = traffic.last_ofmap_words
    middle = traffic.total_words - initial - final

    return _assemble('resa', initial, middle, final,
                     bandwidth_words_per_cycle, computation_cycles)


def planaria_wait(traffic, bandwidth_words_per_cycle, computation_cycles):
    """Everything at the ends: nothing is left in the middle to overlap."""
    assert bandwidth_words_per_cycle > 0

    initial = traffic.read_words
    final = traffic.ofmap_words
    return _assemble('planaria', initial, 0, final,
                     bandwidth_words_per_cycle, computation_cycles)


def _assemble(policy, initial, middle, final, bandwidth, computation):
    initial_cycles = math.ceil(initial / bandwidth)
    middle_cycles = math.ceil(middle / bandwidth) if middle > 0 else 0
    final_cycles = math.ceil(final / bandwidth)
    wait = initial_cycles + max(0, middle_cycles - computation) + final_cycles

    return memory_wait(
        policy=policy,
        initial_words=initial, middle_words=max(0, middle), final_words=final,
        initial_cycles=initial_cycles, middle_cycles=middle_cycles,
        final_cycles=final_cycles, computation_cycles=computation,
        wait_cycles=wait)
