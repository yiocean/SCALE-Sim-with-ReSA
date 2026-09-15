"""Which sub-arrays share a buffer group.

docs/resa_spec.md section 1. Figure 7(a) of the paper draws four dashed boxes,
each holding a 2x2 block of sub-arrays and each fed by its own On-chip Buffer
and Data Router: the groups are the quadrants of the 4x4 grid. The prose never
says so, which is why this is worth writing down.

Membership is still a table rather than a hard-coded rule, for two reasons: the
arbiter must not bake in a geometry, and 'row' is kept as a counterfactual so
Phase 3 can measure how much the fission ranking depends on a detail the paper
only draws.
"""


def quadrant_map(grid_rows, grid_cols, sub_arrays_per_group):
    """Groups are rectangular blocks, as squarish as the counts allow.

    For the paper's 4x4 grid with 4 sub-arrays per group this is the 2x2
    quadrants that Figure 7(a) draws:

        0  1 | 2  3          group 0 = 0, 1, 4, 5
        4  5 | 6  7          group 1 = 2, 3, 6, 7
        -----+-----          group 2 = 8, 9, 12, 13
        8  9 |10 11          group 3 = 10, 11, 14, 15
       12 13 |14 15
    """
    block_rows, block_cols = _squarest_block(grid_rows, grid_cols,
                                             sub_arrays_per_group)
    blocks_per_row = grid_cols // block_cols

    group_of = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            group_of.append((row // block_rows) * blocks_per_row
                            + (col // block_cols))
    return tuple(group_of)


def row_map(grid_rows, grid_cols, sub_arrays_per_group):
    """Groups are consecutive sub-arrays in row-major order.

    A counterfactual, not what Figure 7(a) shows. Kept so the sensitivity of
    the fission ranking to the geometry can be measured instead of assumed.
    For a 4x4 grid with 4 per group, each group is one grid row.
    """
    total = grid_rows * grid_cols
    return tuple(index // sub_arrays_per_group for index in range(total))


GEOMETRIES = {
    'quadrant': quadrant_map,
    'row': row_map,
}


def build_group_map(geometry, grid_rows, grid_cols, sub_arrays_per_group,
                    custom=None):
    """Return a tuple mapping flat sub-array id (row-major) to group id."""
    if geometry == 'custom':
        assert custom is not None, 'GroupGeometry is custom but no map was given'
        group_of = tuple(int(value) for value in custom)
    else:
        assert geometry in GEOMETRIES, (
            'unknown GroupGeometry %r; use one of %s or custom'
            % (geometry, sorted(GEOMETRIES)))
        group_of = GEOMETRIES[geometry](grid_rows, grid_cols,
                                        sub_arrays_per_group)

    validate_group_map(group_of, grid_rows, grid_cols, sub_arrays_per_group)
    return group_of


def validate_group_map(group_of, grid_rows, grid_cols, sub_arrays_per_group):
    """Every sub-array in exactly one group, every group the same size."""
    total = grid_rows * grid_cols
    assert len(group_of) == total, \
        'group map covers %d sub-arrays, expected %d' % (len(group_of), total)

    sizes = {}
    for group_id in group_of:
        sizes[group_id] = sizes.get(group_id, 0) + 1

    expected_groups = total // sub_arrays_per_group
    assert sorted(sizes) == list(range(expected_groups)), \
        'group ids must be 0..%d, got %s' % (expected_groups - 1, sorted(sizes))
    for group_id, size in sizes.items():
        assert size == sub_arrays_per_group, \
            'group %d has %d sub-arrays, expected %d' % (group_id, size,
                                                         sub_arrays_per_group)
    return True


def groups_spanned(group_of, sub_array_ids):
    """The distinct buffer groups a set of sub-arrays touches."""
    return sorted({group_of[sub_array_id] for sub_array_id in sub_array_ids})


def _squarest_block(grid_rows, grid_cols, size):
    """Block shape (rows, cols) with rows*cols == size that tiles the grid."""
    best = None
    for block_rows in range(1, size + 1):
        if size % block_rows:
            continue
        block_cols = size // block_rows
        if grid_rows % block_rows or grid_cols % block_cols:
            continue
        skew = abs(block_rows - block_cols)
        if best is None or skew < best[0]:
            best = (skew, block_rows, block_cols)
    assert best is not None, \
        'no block of %d sub-arrays tiles a %dx%d grid' % (size, grid_rows, grid_cols)
    return best[1], best[2]
