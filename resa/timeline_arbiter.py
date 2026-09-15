"""Buffer-port contention: what a tile's demand matrix costs once groups collide.

docs/resa_plan.md, timeline_arbiter.py. Each buffer group is a banked SRAM
behind one read-and-write port, reached through a 4x4 crossbar. Reads and
writes share that port, so a group's ifmap reads, filter reads and ofmap writes
all draw on the same per-cycle word budget (config
[resa_buffer] OnChipBandwidthWordsPerCycle, split evenly over the groups).

Requests are de-duplicated per cycle within a group, which is the data router
multicasting (docs/resa_spec.md section 8 item 4). The de-duplication is ACROSS
threads: sub-arrays of one thread hold different slices of the logical array
and never ask for the same address in the same cycle, so co-locating a thread's
sub-arrays buys serialisation, not multicast. What pays is two threads of one
group running tiles that share an operand -- and that is the scheduler's
choice, not the geometry's. See docs/resa_spec.md section 1.

Two modes:

  fast      a roofline: max(compute makespan, per-group words / budget).
            O(1) per candidate, so it can re-rank the whole Phase 2 search.
  accurate  wave by wave, cycle by cycle, with exact de-duplication. Vectorised
            by sorting each cycle's addresses rather than looping in Python,
            and chunked along the cycle axis so a wave never has to be
            materialised whole.

The report must carry the gap between them; if it exceeds 10%, fast is not fit
to draw conclusions from (docs/resa_plan.md).
"""

import math

import numpy as np

from resa import cycle_model, port_map
from resa.scheduler import waves

# Cycles per chunk when scanning a wave. Keeps peak memory to roughly
# CHUNK * (ports of the wave) * 4 bytes regardless of how long a tile is.
CHUNK_CYCLES = 32768


class contention_result:
    """What the arbiter measured for one layer."""

    __slots__ = ('total_cycles', 'compute_cycles', 'stall_cycles',
                 'switch_cycles', 'group_words', 'group_busy_cycles',
                 'multicast_savings', 'mode', 'peak_group_load')

    def __init__(self, **kwargs):
        for name in self.__slots__:
            setattr(self, name, kwargs.get(name, 0))

    @property
    def contention_factor(self):
        """Total cycles divided by what the same work costs with free memory."""
        if not self.compute_cycles:
            return 1.0
        return self.total_cycles / self.compute_cycles

    def __repr__(self):
        return ('contention_result(%s: %d cycles = %d compute + %d stall + %d switch,'
                ' %.2fx)' % (self.mode, self.total_cycles, self.compute_cycles,
                             self.stall_cycles, self.switch_cycles,
                             self.contention_factor))


# --------------------------------------------------------------------------
# Fast mode
# --------------------------------------------------------------------------

def edge_sub_arrays_per_group(fission_obj, group_map, num_groups, dataflow='ws'):
    """How many distinct sub-arrays draw on each group, counted over all threads.

    Only edge sub-arrays talk to a buffer, so this is well below the group's
    membership for the larger fissions: a 4x4 thread reaches group 0 through
    three sub-arrays (its top-left corner, the one below it and the one to its
    right), while a 1x1 fission puts four independent threads on the same group.
    """
    counts = np.zeros(num_groups, dtype=np.int64)
    for group_id in range(num_groups):
        owners = set()
        for thread_id in range(fission_obj.num_threads):
            groups = port_map.port_groups(fission_obj, group_map, thread_id,
                                          dataflow)
            edges = port_map.port_sub_arrays(fission_obj, thread_id, dataflow)
            for port_group, port_owner in zip(groups, edges):
                owners.update(port_owner[port_group == group_id].tolist())
        counts[group_id] = len(owners)
    return counts

def fast_layer_estimate(resa_cfg, fission_obj, dataflow, tile_shapes,
                        window_size, valid_ifmap_total, compute_makespan,
                        switch_cycles=0):
    """Roofline estimate for one layer under one configuration.

    `tile_shapes` is [(m_len, n_len, count)] as produced by tile_search.

    A tile's accesses are split over the groups in proportion to how many of
    its operand ports each group owns. That is exact for filter and ofmap,
    whose ports carry equal numbers of requests, and an approximation for
    ifmap, whose columns differ slightly when some convolution windows fall
    outside the input.
    """
    num_groups, _, _ = resa_cfg.get_buffer_group_layout()
    budget = resa_cfg.get_group_port_budget()
    group_map = resa_cfg.get_group_map()
    arr_row, arr_col = fission_obj.logical_dims

    words = np.zeros(num_groups, dtype=np.float64)
    total_m = sum(m_len * count for m_len, _, count in tile_shapes)

    for thread_id in range(fission_obj.num_threads):
        share = port_map.port_group_counts(fission_obj, group_map, thread_id,
                                           num_groups, dataflow)
        # tiles are handed round-robin to threads, so each thread carries
        # roughly its share of every shape
        for m_len, n_len, count in tile_shapes:
            per_thread = count / fission_obj.num_threads
            valid = valid_ifmap_total * (m_len / total_m) / max(count, 1) \
                if total_m else m_len * window_size
            ifmap, filt, ofmap = cycle_model.access_counts(
                dataflow, m_len, n_len, window_size, arr_row, arr_col,
                valid_ifmap=min(int(round(valid)), m_len * window_size))
            words += per_thread * (
                ifmap * share[0] / max(share[0].sum(), 1)
                + filt * share[1] / max(share[1].sum(), 1)
                + ofmap * share[2] / max(share[2].sum(), 1))

    bandwidth_cycles = int(math.ceil(words.max() / budget)) if budget else 0

    if resa_cfg.get_port_model() in ('subarray_serial', 'both'):
        # Only EDGE sub-arrays ever talk to a buffer: the leftmost column takes
        # ifmap, the top row takes filter, the bottom row drains ofmap. So the
        # number of sub-arrays sharing a port is a property of the fission, not
        # just of how many sub-arrays sit in the group -- a 4x4 thread puts
        # three of its sixteen sub-arrays on group 0, a 1x1 fission puts four
        # separate threads there.
        _, _, ports = resa_cfg.get_buffer_group_layout()
        sharers = edge_sub_arrays_per_group(fission_obj, group_map, num_groups,
                                            dataflow)
        serial_cycles = int(math.ceil(compute_makespan * sharers.max() / ports))
        if resa_cfg.get_port_model() == 'subarray_serial':
            bandwidth_cycles = serial_cycles
        else:
            bandwidth_cycles = max(bandwidth_cycles, serial_cycles)

    total = max(compute_makespan, bandwidth_cycles) + switch_cycles

    return contention_result(
        mode='fast',
        total_cycles=total,
        compute_cycles=compute_makespan,
        stall_cycles=max(0, total - compute_makespan - switch_cycles),
        switch_cycles=switch_cycles,
        group_words=words.astype(np.int64),
        peak_group_load=float(words.max() / max(compute_makespan, 1)))


# --------------------------------------------------------------------------
# Accurate mode
# --------------------------------------------------------------------------

def _wave_group_columns(resa_cfg, fission_obj, wave, demand_for):
    """Per group, the address columns every thread of the wave contributes.

    Returns {group_id: [(matrix, column indices, sub-array ids)]} plus the wave
    length. The sub-array ids are what the subarray_serial port model counts.
    """
    num_groups, _, _ = resa_cfg.get_buffer_group_layout()
    group_map = resa_cfg.get_group_map()

    columns = {group_id: [] for group_id in range(num_groups)}
    wave_length = 0
    for job in wave:
        matrices = demand_for(job)
        wave_length = max(wave_length, matrices[0].shape[0])
        ports = port_map.port_groups(fission_obj, group_map, job.thread_id,
                                     job.dataflow)
        owners = port_map.port_sub_arrays(fission_obj, job.thread_id, job.dataflow)
        for matrix, port_group, port_owner in zip(matrices, ports, owners):
            for group_id in np.unique(port_group):
                selected = np.flatnonzero(port_group == group_id)
                columns[group_id].append((matrix, selected, port_owner[selected]))
    return columns, wave_length


def _distinct_per_cycle(block):
    """Number of distinct non-null addresses in each row of `block`.

    Sorting each row and counting value changes is what makes the accurate mode
    affordable: it is one vectorised sort instead of a Python loop per cycle.
    """
    if block.shape[1] == 0:
        return np.zeros(block.shape[0], dtype=np.int64)

    ordered = np.sort(block, axis=1)
    changed = np.empty(ordered.shape, dtype=bool)
    changed[:, 0] = True
    np.not_equal(ordered[:, 1:], ordered[:, :-1], out=changed[:, 1:])
    valid = ordered != -1
    return np.count_nonzero(changed & valid, axis=1).astype(np.int64)


def _active_sub_arrays(block, owners):
    """How many distinct sub-arrays have a live request in each cycle.

    The subarray_serial reading of "single read-and-write port": the crossbar
    picks a sub-array per cycle, so four sub-arrays of one group take four
    cycles however few words each of them wants.
    """
    live = block != -1
    unique_owners = np.unique(owners)
    active = np.zeros(block.shape[0], dtype=np.int64)
    for owner in unique_owners:
        active += live[:, owners == owner].any(axis=1)
    return active


def accurate_layer(resa_cfg, fission_obj, jobs, demand_for, switch_cycles=0):
    """Cycle-accurate contention over all waves of a layer.

    A wave runs in lockstep: every thread in it advances one cycle per step,
    and when a group cannot serve a step's requests within its budget the whole
    wave waits. That is the same schedule as list scheduling whenever the tiles
    are equal sized, which is the usual case.
    """
    num_groups, _, ports_per_group = resa_cfg.get_buffer_group_layout()
    budget = resa_cfg.get_group_port_budget()
    port_model = resa_cfg.get_port_model()

    compute_cycles = 0
    stall_cycles = 0
    group_words = np.zeros(num_groups, dtype=np.int64)
    group_busy = np.zeros(num_groups, dtype=np.int64)
    multicast_savings = 0
    peak_load = 0

    for wave in waves(jobs, fission_obj.num_threads):
        columns, wave_length = _wave_group_columns(resa_cfg, fission_obj, wave,
                                                   demand_for)
        compute_cycles += wave_length

        for start in range(0, wave_length, CHUNK_CYCLES):
            stop = min(start + CHUNK_CYCLES, wave_length)
            span = stop - start
            service = np.ones(span, dtype=np.int64)

            for group_id, pieces in columns.items():
                if not pieces:
                    continue
                blocks = []
                owners = []
                for matrix, selected, port_owner in pieces:
                    rows = matrix.shape[0]
                    if start >= rows:
                        continue
                    piece = matrix[start:min(stop, rows)][:, selected]
                    if piece.shape[0] < span:      # a shorter tile in this wave
                        piece = np.concatenate(
                            (piece, np.full((span - piece.shape[0], piece.shape[1]),
                                            -1, dtype=piece.dtype)))
                    blocks.append(piece)
                    owners.append(port_owner)
                if not blocks:
                    continue

                block = np.concatenate(blocks, axis=1)
                raw = np.count_nonzero(block != -1, axis=1).astype(np.int64)
                distinct = _distinct_per_cycle(block)

                multicast_savings += int((raw - distinct).sum())
                group_words[group_id] += int(distinct.sum())
                peak_load = max(peak_load, int(distinct.max(initial=0)))

                needed = np.ones(span, dtype=np.int64)
                if port_model in ('bandwidth', 'both'):
                    np.maximum(needed,
                               np.ceil(distinct / budget).astype(np.int64),
                               out=needed)
                if port_model in ('subarray_serial', 'both'):
                    active = _active_sub_arrays(block, np.concatenate(owners))
                    np.maximum(needed,
                               np.ceil(active / ports_per_group).astype(np.int64),
                               out=needed)

                group_busy[group_id] += int(np.count_nonzero(distinct))
                np.maximum(service, needed, out=service)

            stall_cycles += int((service - 1).sum())

    total = compute_cycles + stall_cycles + switch_cycles
    return contention_result(
        mode='accurate',
        total_cycles=total,
        compute_cycles=compute_cycles,
        stall_cycles=stall_cycles,
        switch_cycles=switch_cycles,
        group_words=group_words,
        group_busy_cycles=group_busy,
        multicast_savings=multicast_savings,
        peak_group_load=peak_load)
