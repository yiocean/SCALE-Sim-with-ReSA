"""Which buffer group serves each operand port of a logical array.

Operands only enter a systolic array at its edges; inside, data moves PE to PE.
So the demand matrix columns are exactly the edge injections, and each column
belongs to whichever sub-array owns that edge position:

    left edge    one port per array ROW  -> sub-array (origin_r + i // sub_rows, origin_c)
    top edge     one port per array COL  -> sub-array (origin_r, origin_c + j // sub_cols)
    bottom edge  one port per array COL  -> sub-array (origin_r + Fr - 1, origin_c + j // sub_cols)

Which operand uses which edge depends on the dataflow, and IS is not like the
other two: it holds the ifmap stationary and streams the filters, so its ifmap
arrives from the top and its filters from the left -- the mirror image of WS.
The demand matrix widths say so directly (IS's ifmap matrix is arr_col wide,
its filter matrix arr_row wide), which is what
tests/test_arbiter.py::test_port_map_widths_match_the_demand_matrices checks.

The NoC is unidirectional (left-to-right, top-to-bottom), so nothing enters
from the right or the bottom. The bottom-edge drain is the logical array's
bottom, not the physical array's: each sub-array has its own accumulator and
can send results straight back to its buffer (spec section 3, Figure 8).
"""

import numpy as np

# (ifmap edge, filter edge, ofmap edge) per dataflow.
OPERAND_EDGES = {
    'ws': ('left', 'top', 'bottom'),
    'is': ('top', 'left', 'bottom'),
    'os': ('left', 'top', 'bottom'),
}


def _edge_sub_arrays(fission_obj, thread_id, edge):
    """Flat sub-array id behind each port of one edge of a thread."""
    origin_row, origin_col = fission_obj.thread_origin(thread_id)
    grid_cols = fission_obj.grid_cols
    sub_rows = fission_obj.sub_array_rows
    sub_cols = fission_obj.sub_array_cols
    arr_row, arr_col = fission_obj.logical_dims

    def flat(grid_row, grid_col):
        return grid_row * grid_cols + grid_col

    if edge == 'left':
        return np.array([flat(origin_row + port // sub_rows, origin_col)
                         for port in range(arr_row)], dtype=np.int16)
    if edge == 'top':
        return np.array([flat(origin_row, origin_col + port // sub_cols)
                         for port in range(arr_col)], dtype=np.int16)
    if edge == 'bottom':
        return np.array([flat(origin_row + fission_obj.rows - 1,
                              origin_col + port // sub_cols)
                         for port in range(arr_col)], dtype=np.int16)
    raise KeyError('unknown edge: %r' % (edge,))


def port_sub_arrays(fission_obj, thread_id, dataflow='ws'):
    """(ifmap, filter, ofmap) arrays giving the sub-array behind each port."""
    assert dataflow in OPERAND_EDGES, 'unknown dataflow: %r' % (dataflow,)
    return tuple(_edge_sub_arrays(fission_obj, thread_id, edge)
                 for edge in OPERAND_EDGES[dataflow])


def port_groups(fission_obj, group_map, thread_id, dataflow='ws'):
    """(ifmap, filter, ofmap) arrays giving the buffer group of each port."""
    return tuple(np.array([group_map[sub_array] for sub_array in owners],
                          dtype=np.int16)
                 for owners in port_sub_arrays(fission_obj, thread_id, dataflow))


def thread_groups(fission_obj, group_map, thread_id, dataflow='ws'):
    """The distinct buffer groups a thread draws on, sorted."""
    groups = port_groups(fission_obj, group_map, thread_id, dataflow)
    return sorted({int(value) for array in groups for value in array})


def port_group_counts(fission_obj, group_map, thread_id, num_groups,
                      dataflow='ws'):
    """How many ports of each operand each group owns, as (3, num_groups).

    Row order is ifmap, filter, ofmap. This is what the fast estimate needs:
    the share of a tile's accesses that lands on each group.
    """
    counts = np.zeros((3, num_groups), dtype=np.int64)
    for index, ports in enumerate(port_groups(fission_obj, group_map,
                                              thread_id, dataflow)):
        for group_id in ports:
            counts[index, group_id] += 1
    return counts
