"""Algorithm 1: offline search for the best tiling, dataflow and fission.

docs/resa_spec.md section 4. Two deliberate differences from the paper, both
recorded in section 8:

  * the IC loop is disabled (item 1) -- splitting K would need cross-sub-array
    partial-sum reduction, which SCALE-Sim does not model, so every candidate
    sees the whole conv window;
  * (ow, oh) are enumerated exactly as the paper does, but only as a generator
    of candidate M tile *sizes* (item 9). M is a pure time dimension in
    SCALE-Sim's GEMM mapping, so a strided (oh x ow) tile and a contiguous M
    range of the same size produce identical cycle counts.

Examine() is arithmetic on the size laws in resa/cycle_model.py, never a
materialised demand matrix: the space is ~4k points per layer after
de-duplicating tile sizes, and hundreds of thousands across the four
workloads. The laws are validated against real SCALE-Sim output in
tests/test_cycle_model.py, and search_layer(verify=True) re-runs the winning
configuration for real and asserts the prediction was right -- so a law that
breaks down at some boundary fails loudly on that layer instead of quietly
producing a wrong profiling table.
"""

import csv
import math
import os

from resa import cycle_model
from resa.fission import assert_uniform, fission_space
from resa.subarray_group import build_operand_matrices, subarray_group
from resa.tensor_split import layer_tiling, split_operands

# Fields of the paper's profiling table (docs/resa_spec.md section 5), followed
# by the columns this reimplementation needs to make a result reproducible.
PROFILING_FIELDS = [
    'model', 'layer_id', 'layer_name', 'tiles',
    'Ow', 'Oh', 'kw', 'kh', 'C', 'F', 'R', 'S', 'dataflow',
    'm_tile_size', 'n_tile_size', 'threads',
    'tile_cycles_max', 'tile_cycles_total', 'predicted_cycles',
    'monolithic_ws_cycles', 'speedup_vs_monolithic_ws',
]


# One kept tiling inside a ranking cell, plus the cell metadata Phase 3 needs
# to tell an honest re-rank from one that hit the pruning boundary.
RANKING_FIELDS = [
    'model', 'layer_id', 'layer_name', 'R', 'S', 'dataflow',
    'rank_in_cell', 'm_tile_size', 'n_tile_size', 'tiles', 'threads',
    'tile_cycles_max', 'tile_cycles_total', 'predicted_cycles',
    'ratio_to_cell_best', 'cell_candidates', 'cell_truncated',
]


class ranking_cell:
    """The tilings kept for one (fission, dataflow) pair.

    Phase 3 re-ranks these once buffer-port contention is modelled. Keeping
    only the compute-only winner would be wrong: contention changes which
    tiling wins inside a cell, not just which cell wins. Small tiles look good
    to the compute-only objective (threads stay full) and bad under contention
    (more operand replay, more port pressure), so the two objectives disagree
    exactly along the axis being pruned.

    `truncated` says the keep limit, not the margin, decided what was dropped:
    if Phase 3's winner is the last kept entry of a truncated cell, the answer
    sat on the pruning boundary and the search must be re-run with a larger
    RankingTilingsPerCell.
    """

    __slots__ = ('fission_key', 'dataflow', 'kept', 'candidates', 'truncated')

    def __init__(self, fission_key, dataflow):
        self.fission_key = fission_key
        self.dataflow = dataflow
        self.kept = []
        self.candidates = 0
        self.truncated = False

    def add(self, record):
        self.candidates += 1
        self.kept.append(record)

    def prune(self, keep, margin_percent):
        """Keep the best `keep` tilings that are within `margin_percent` of the best."""
        self.kept.sort(key=lambda record: (record['predicted_cycles'],
                                           record['tile_cycles_total'],
                                           record['tiles']))
        best = self.kept[0]['predicted_cycles']
        limit = best * (1.0 + margin_percent / 100.0)
        within = [record for record in self.kept
                  if record['predicted_cycles'] <= limit]
        self.truncated = len(within) > keep
        self.kept = within[:keep]
        for rank, record in enumerate(self.kept):
            record['rank_in_cell'] = rank
            record['ratio_to_cell_best'] = record['predicted_cycles'] / best

    @property
    def best_cycles(self):
        return self.kept[0]['predicted_cycles']


class profiling_entry:
    """One row of the profiling table: the winning configuration for a layer."""

    __slots__ = tuple(PROFILING_FIELDS)

    def __init__(self, **kwargs):
        for field in PROFILING_FIELDS:
            setattr(self, field, kwargs.get(field))

    def as_row(self):
        return {field: getattr(self, field) for field in PROFILING_FIELDS}

    def __repr__(self):
        return ('profiling_entry(%s L%s %s fission %dx%d, %d tiles, %d cycles)'
                % (self.model, self.layer_id, self.dataflow, self.R, self.S,
                   self.tiles, self.predicted_cycles))


def exponential_candidates(limit):
    """The paper's 1, 2, 4, ... enumeration: min(2**i, limit) for i in 0..ceil(lg limit).

    Returns [(value, exponent)], de-duplicated, smallest first.
    """
    assert limit >= 1
    count = math.ceil(math.log2(limit)) + 1 if limit > 1 else 1
    seen = {}
    for exponent in range(count):
        value = min(2 ** exponent, limit)
        seen.setdefault(value, exponent)
    return sorted((value, exponent) for value, exponent in seen.items())


def m_tile_candidates(ofmap_rows, ofmap_cols):
    """Candidate M tile sizes, with the (oh, ow) pair that generated each.

    The paper's nested (ow, oh) enumeration; the product is what reaches the
    tiling, the pair is kept only so the profiling table can report the same
    Ow / Oh fields the paper does.
    """
    candidates = {}
    for ow, _ in exponential_candidates(ofmap_cols):
        for oh, _ in exponential_candidates(ofmap_rows):
            candidates.setdefault(oh * ow, (oh, ow))
    return sorted((size, pair) for size, pair in candidates.items())


def n_tile_candidates(num_filters):
    """Candidate N tile sizes, i.e. the paper's oc."""
    return [value for value, _ in exponential_candidates(num_filters)]


def tile_shapes(num_rows, num_cols, m_tile_size, n_tile_size):
    """The at most four distinct tile shapes a uniform tiling produces.

    Returns [(m_len, n_len, count)]. Enumerating tiles one by one would be far
    too slow -- a layer can have hundreds of thousands of them -- but a uniform
    tiling only ever has full tiles plus a short last row and column.
    """
    num_m = math.ceil(num_rows / m_tile_size)
    num_n = math.ceil(num_cols / n_tile_size)
    m_last = num_rows - (num_m - 1) * m_tile_size
    n_last = num_cols - (num_n - 1) * n_tile_size

    m_groups = [(m_tile_size, num_m - 1), (m_last, 1)]
    n_groups = [(n_tile_size, num_n - 1), (n_last, 1)]

    shapes = []
    for m_len, m_count in m_groups:
        if m_count == 0:
            continue
        for n_len, n_count in n_groups:
            if n_count == 0:
                continue
            shapes.append((m_len, n_len, m_count * n_count))
    return shapes


def examine(dataflow, arr_row, arr_col, num_threads,
            num_rows, num_cols, window_size, m_tile_size, n_tile_size):
    """Algorithm 1's Examine(): estimated cycles for one candidate.

    Returns (estimated cycles, tile count, largest tile cycles, total tile cycles).

    The estimate is the identical-machines makespan bound

        max(largest tile, ceil(total tile cycles / threads))

    which is exact when the tiling divides evenly and otherwise slightly
    optimistic; docs/resa_spec.md section 8, item 10. The real
    list-scheduling makespan is Phase 3's job.
    """
    total_cycles = 0
    max_cycles = 0
    num_tiles = 0

    for m_len, n_len, count in tile_shapes(num_rows, num_cols,
                                           m_tile_size, n_tile_size):
        cycles = cycle_model.demand_rows(dataflow, m_len, n_len, window_size,
                                         arr_row, arr_col)
        total_cycles += cycles * count
        max_cycles = max(max_cycles, cycles)
        num_tiles += count

    estimate = max(max_cycles, math.ceil(total_cycles / num_threads))
    return estimate, num_tiles, max_cycles, total_cycles


def search_layer(resa_cfg, topo_obj, layer_id, model_name='',
                 space=None, dataflows=cycle_model.DATAFLOWS, verify=True,
                 return_ranking=False):
    """Search one layer. Returns (winning entry, number of candidates examined).

    With verify=True the winning configuration is run through SCALE-Sim for
    real and the predicted cycle count of every distinct tile shape is checked
    against what comes back.

    With return_ranking=True a third element is returned: a ranking_cell for
    every (fission, dataflow) pair, each holding the best few tilings of that
    cell rather than only its winner. The compute-only winner is usually a few
    percent ahead of the runner-up (docs/resa_spec.md section 8, item 10), so
    Phase 3 re-ranks once contention is modelled; how many tilings survive is
    set by [resa_search] in the config.

    Ties on the estimate are broken by, in order: less total tile work, then
    fewer tiles, then the LARGER logical array. The last is deliberate: the
    objective cannot see SRAM traffic, and a larger array replays operands
    less, so on a tie the configuration whose hidden cost is lower wins.
    """
    if space is None:
        space = fission_space(resa_cfg)

    num_rows, num_cols, window_size = cycle_model.gemm_dims_from_layer(topo_obj,
                                                                       layer_id)
    ofmap_rows, ofmap_cols = topo_obj.get_layer_ofmap_dims(layer_id)
    filter_rows, filter_cols = topo_obj.get_layer_filter_dims(layer_id)
    channels = int(topo_obj.get_layer_num_channels(layer_id))

    best = None
    examined = 0
    ranking = {}
    for m_tile_size, (oh, ow) in m_tile_candidates(int(ofmap_rows), int(ofmap_cols)):
        for n_tile_size in n_tile_candidates(num_cols):
            for dataflow in dataflows:
                for fission_obj in space.all():
                    arr_row, arr_col = fission_obj.logical_dims
                    estimate, tiles, max_cycles, total_cycles = examine(
                        dataflow, arr_row, arr_col, fission_obj.num_threads,
                        num_rows, num_cols, window_size,
                        m_tile_size, n_tile_size)
                    examined += 1

                    if return_ranking:
                        slot = ((fission_obj.rows, fission_obj.cols), dataflow)
                        if slot not in ranking:
                            ranking[slot] = ranking_cell(slot[0], dataflow)
                        ranking[slot].add(dict(
                            m_tile_size=m_tile_size, n_tile_size=n_tile_size,
                            tiles=tiles, threads=fission_obj.num_threads,
                            tile_cycles_max=max_cycles,
                            tile_cycles_total=total_cycles,
                            predicted_cycles=estimate))

                    key = (estimate, total_cycles, tiles,
                           -fission_obj.sub_arrays_per_thread)
                    if best is None or key < best[0]:
                        best = (key, dict(
                            fission=fission_obj, dataflow=dataflow,
                            m_tile_size=m_tile_size, n_tile_size=n_tile_size,
                            oh=oh, ow=ow, tiles=tiles,
                            tile_cycles_max=max_cycles,
                            tile_cycles_total=total_cycles,
                            predicted_cycles=estimate))

    assert best is not None, 'empty search space for layer %d' % layer_id
    won = best[1]
    fission_obj = won['fission']

    # A layer runs on one fission; the paper forbids mixing them at any instant.
    assert_uniform([fission_obj] * fission_obj.num_threads)

    monolithic = space.monolithic()
    monolithic_arr = monolithic.logical_dims
    monolithic_cycles = cycle_model.demand_rows('ws', num_rows, num_cols,
                                                window_size, *monolithic_arr)

    entry = profiling_entry(
        model=model_name,
        layer_id=layer_id,
        layer_name=topo_obj.topo_arrays[layer_id][0],
        tiles=won['tiles'],
        Ow=won['ow'], Oh=won['oh'],
        kw=int(filter_cols), kh=int(filter_rows),
        C=channels, F=won['n_tile_size'],
        R=fission_obj.rows, S=fission_obj.cols,
        dataflow=won['dataflow'],
        m_tile_size=won['m_tile_size'], n_tile_size=won['n_tile_size'],
        threads=fission_obj.num_threads,
        tile_cycles_max=won['tile_cycles_max'],
        tile_cycles_total=won['tile_cycles_total'],
        predicted_cycles=won['predicted_cycles'],
        monolithic_ws_cycles=monolithic_cycles,
        speedup_vs_monolithic_ws=monolithic_cycles / won['predicted_cycles'],
    )

    if verify:
        verify_entry(resa_cfg, topo_obj, layer_id, entry)

    if return_ranking:
        keep, margin = resa_cfg.get_ranking_limits()
        for cell in ranking.values():
            cell.prune(keep, margin)
        return entry, examined, ranking
    return entry, examined


def verify_entry(resa_cfg, topo_obj, layer_id, entry):
    """Run the winning configuration for real and check the law predicted it.

    Every distinct tile shape is built and compared, so a law that fails at
    some boundary condition blows up on the layer that hit it rather than
    silently corrupting the profiling table.
    """
    scale_cfg = resa_cfg.get_scale_config()
    ifmap_op_mat, filter_op_mat, ofmap_op_mat = \
        build_operand_matrices(scale_cfg, topo_obj, layer_id)

    num_rows, num_cols, window_size = cycle_model.gemm_dims_from_operands(
        ifmap_op_mat, filter_op_mat)
    arr_row = entry.R * resa_cfg.get_sub_array_dims()[0]
    arr_col = entry.S * resa_cfg.get_sub_array_dims()[1]

    tiling = layer_tiling(num_rows, num_cols,
                          m_tile_size=entry.m_tile_size,
                          n_tile_size=entry.n_tile_size)
    assert tiling.num_tiles == entry.tiles, \
        'tile count disagrees with the search: %d vs %d' % (tiling.num_tiles,
                                                            entry.tiles)

    checked = set()
    for tile_obj in tiling.tiles():
        shape = (tile_obj.m_len, tile_obj.n_len)
        if shape in checked:
            continue
        checked.add(shape)

        parts = split_operands(ifmap_op_mat, filter_op_mat, ofmap_op_mat, tile_obj)
        group = subarray_group(scale_cfg, arr_row, arr_col, dataflow=entry.dataflow)
        produced = group.run(*parts)[0].shape[0]
        predicted = cycle_model.demand_rows(entry.dataflow, tile_obj.m_len,
                                            tile_obj.n_len, window_size,
                                            arr_row, arr_col)
        assert produced == predicted, (
            'size law failed on %s layer %d, tile %dx%d under %s at %dx%d: '
            'predicted %d, SCALE-Sim produced %d'
            % (entry.model, layer_id, tile_obj.m_len, tile_obj.n_len,
               entry.dataflow, arr_row, arr_col, predicted, produced))
    return True


def search_model(resa_cfg, topo_obj, model_name, verify=True,
                 dataflows=cycle_model.DATAFLOWS, progress=None,
                 return_ranking=False):
    """Search every layer of one workload.

    Returns a list of profiling entries, or (entries, rankings) with
    return_ranking=True, where rankings[layer_id] is the ranking of that layer.
    """
    space = fission_space(resa_cfg)
    entries = []
    rankings = {}
    for layer_id in range(topo_obj.get_num_layers()):
        result = search_layer(resa_cfg, topo_obj, layer_id,
                              model_name=model_name, space=space,
                              dataflows=dataflows, verify=verify,
                              return_ranking=return_ranking)
        if return_ranking:
            entry, examined, ranking = result
            rankings[layer_id] = ranking
        else:
            entry, examined = result
        entries.append(entry)
        if progress is not None:
            progress(entry, examined)

    if return_ranking:
        return entries, rankings
    return entries


def write_ranking_table(entries, rankings, path):
    """Write the pruned search ranking Phase 3 re-ranks from.

    One row per kept tiling per (fission, dataflow) cell per layer.
    """
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    by_layer = {(entry.model, entry.layer_id): entry for entry in entries}

    with open(path, 'w', newline='') as handle:
        handle.write(
            '# RESA search ranking -- based on our reimplementation of ReSA\n'
            '# (Lee & Yeh, ACM TACO 21(3), 2024). Compute-only estimates: no\n'
            '# memory contention, no reconfiguration penalty.\n'
            '# Phase 3 re-ranks these rows once buffer-port contention is\n'
            '# modelled. The space here is already pruned by the compute-only\n'
            '# objective (docs/resa_spec.md section 8, item 10): if the row\n'
            '# Phase 3 picks is the last kept row of a cell with\n'
            '# cell_truncated = True, the answer sat on the pruning boundary\n'
            '# and the search must be re-run with a larger\n'
            '# [resa_search] RankingTilingsPerCell.\n')
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS)
        writer.writeheader()
        for (model, layer_id), ranking in sorted(rankings.items()):
            entry = by_layer[(model, layer_id)]
            for (fission_key, dataflow), cell in sorted(ranking.items()):
                for record in cell.kept:
                    row = dict(record)
                    row.update(model=model, layer_id=layer_id,
                               layer_name=entry.layer_name,
                               R=fission_key[0], S=fission_key[1],
                               dataflow=dataflow,
                               cell_candidates=cell.candidates,
                               cell_truncated=cell.truncated)
                    writer.writerow({field: row[field] for field in RANKING_FIELDS})
    return path


def write_profiling_table(entries, path):
    """Write the profiling table, with the provenance header the plan requires."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, 'w', newline='') as handle:
        handle.write(
            '# RESA profiling table -- based on our reimplementation of ReSA\n'
            '# (Lee & Yeh, ACM TACO 21(3), 2024). Numbers are NOT claimed to\n'
            '# match the paper. Modelling decisions that deviate from it are\n'
            '# listed in docs/resa_spec.md section 8; the ones that shape this\n'
            '# table are item 1 (K splitting disabled), item 9 (M tile size\n'
            '# only, no 2D shape) and item 10 (makespan estimate used as the\n'
            '# search objective). Cycle counts here are compute-only: no\n'
            '# memory contention, no reconfiguration penalty (Phase 3/4).\n'
            '# Ow / Oh are tile sizes, not layer sizes (spec section 4).\n'
            '# R / S are the fission height and width.\n')
        writer = csv.DictWriter(handle, fieldnames=PROFILING_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.as_row())
    return path
