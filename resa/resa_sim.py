"""Top-level driver: run a layer or a whole model under a chosen configuration.

Ties together tensor_split (tiles), scheduler (tiles onto threads) and
timeline_arbiter (what the buffer ports cost). Phase 2's profiling table says
which configuration to run; this says what it actually costs once the four
single-ported buffer groups are in the way.

Demand matrices are built wave by wave and dropped as soon as the wave is
scored, so a layer never has to exist in memory whole -- UNet_maestro's worst
layer would need gigabytes (docs/resa_phase0_notes.md section 6.2).
"""

import numpy as np

from resa import cycle_model, timeline_arbiter
from resa.fission import fission_space
from resa.scheduler import (balanced_makespan_bound, list_schedule,
                            wave_makespan, waves)
from resa.subarray_group import build_operand_matrices, subarray_group
from resa.tensor_split import layer_tiling, split_operands


class layer_result:
    """One layer under one configuration, compute-only and contended."""

    __slots__ = ('model', 'layer_id', 'layer_name', 'fission', 'dataflow',
                 'm_tile_size', 'n_tile_size', 'tiles', 'threads',
                 'compute_makespan', 'balanced_bound', 'contention', 'mode')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name))

    @property
    def total_cycles(self):
        return self.contention.total_cycles

    def __repr__(self):
        return ('layer_result(%s L%s %s %dx%d: %d cycles, %.2fx contention)'
                % (self.model, self.layer_id, self.dataflow,
                   self.fission.rows, self.fission.cols,
                   self.total_cycles, self.contention.contention_factor))


class layer_operands:
    """Whole-layer operand matrices plus the per-row valid ifmap counts."""

    def __init__(self, resa_cfg, topo_obj, layer_id):
        self.matrices = build_operand_matrices(resa_cfg.get_scale_config(),
                                               topo_obj, layer_id)
        self.m, self.n, self.k = cycle_model.gemm_dims_from_operands(
            self.matrices[0], self.matrices[1])
        rows = cycle_model.valid_ifmap_rows(self.matrices[0])
        self.valid_prefix = np.concatenate(([0], np.cumsum(rows)))

    def valid_ifmap(self, m_start, m_len):
        return int(self.valid_prefix[m_start + m_len] - self.valid_prefix[m_start])

    @property
    def valid_total(self):
        return int(self.valid_prefix[-1])


def evaluate_layer(resa_cfg, operands, fission_obj, dataflow,
                   m_tile_size, n_tile_size, mode='accurate',
                   switch_cycles=0, model='', layer_id=0, layer_name=''):
    """Score one (fission, dataflow, tiling) on one layer."""
    assert mode in ('fast', 'accurate'), 'mode must be fast or accurate'

    arr_row, arr_col = fission_obj.logical_dims
    tiling = layer_tiling(operands.m, operands.n,
                          m_tile_size=m_tile_size, n_tile_size=n_tile_size)
    tiles = tiling.tiles()

    def tile_cycles(tile_obj):
        return cycle_model.demand_rows(dataflow, tile_obj.m_len, tile_obj.n_len,
                                       operands.k, arr_row, arr_col)

    jobs, _ = list_schedule(tiles, tile_cycles, fission_obj.num_threads, dataflow)
    compute_makespan = wave_makespan(jobs, fission_obj.num_threads)
    bound = balanced_makespan_bound([tile_cycles(t) for t in tiles],
                                    fission_obj.num_threads)

    if mode == 'fast':
        shapes = _shapes_of(tiling)
        contention = timeline_arbiter.fast_layer_estimate(
            resa_cfg, fission_obj, dataflow, shapes, operands.k,
            operands.valid_total, compute_makespan, switch_cycles=switch_cycles)
    else:
        contention = timeline_arbiter.accurate_layer(
            resa_cfg, fission_obj, jobs,
            _demand_builder(resa_cfg, operands, arr_row, arr_col, dataflow),
            switch_cycles=switch_cycles)

    return layer_result(
        model=model, layer_id=layer_id, layer_name=layer_name,
        fission=fission_obj, dataflow=dataflow,
        m_tile_size=tiling.m_tile_size, n_tile_size=tiling.n_tile_size,
        tiles=tiling.num_tiles, threads=fission_obj.num_threads,
        compute_makespan=compute_makespan, balanced_bound=bound,
        contention=contention, mode=mode)


def _shapes_of(tiling):
    from resa.tile_search import tile_shapes
    return tile_shapes(tiling.num_operand_rows, tiling.num_filters,
                       tiling.m_tile_size, tiling.n_tile_size)


def _demand_builder(resa_cfg, operands, arr_row, arr_col, dataflow):
    """A demand_for(job) that builds a tile's matrices on demand and caches
    only what the current wave needs."""
    scale_cfg = resa_cfg.get_scale_config()
    cache = {}

    def demand_for(job):
        key = (job.tile.m_start, job.tile.m_len, job.tile.n_start, job.tile.n_len)
        if key not in cache:
            if len(cache) > 4 * 16:      # never hold more than a wave or two
                cache.clear()
            group = subarray_group(scale_cfg, arr_row, arr_col, dataflow=dataflow)
            cache[key] = group.run(*split_operands(*operands.matrices, job.tile))
        return cache[key]

    return demand_for
