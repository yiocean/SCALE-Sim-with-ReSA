"""Tile -> thread assignment and the wave structure the arbiter simulates.

docs/resa_plan.md, scheduler.py. Threads are the logical arrays a fission
produces; tiles are independent, so this is scheduling independent jobs on
identical machines.

Two things here are modelling decisions rather than paper content, both in
docs/resa_spec.md section 8 item 5:

  * tiles are handed out in emission order to the earliest-free thread (list
    scheduling), and
  * the arbiter consumes the result in *waves*: a barrier after each batch of
    at most num_threads tiles. For the usual case of equal-sized tiles the two
    are the same schedule. They differ only on the ragged last tiles, where the
    wave model is slightly pessimistic.

Thread placement is deliberately a parameter, not a constant. Multicast pays
when threads of one buffer group run tiles that share an operand
(docs/resa_spec.md section 1), so which tile lands on which thread is a real
choice; nothing here bakes in row-major order even though that is the default.
"""

import math


class tile_job:
    """One tile placed on one thread."""

    __slots__ = ('tile', 'thread_id', 'dataflow', 'cycles', 'start', 'end',
                 'switch_penalty')

    def __init__(self, tile, thread_id, dataflow, cycles):
        self.tile = tile
        self.thread_id = thread_id
        self.dataflow = dataflow
        self.cycles = cycles
        self.start = 0
        self.end = cycles
        self.switch_penalty = 0

    def __repr__(self):
        return ('tile_job(%r on thread %d, %s, %d cycles at %d)'
                % (self.tile, self.thread_id, self.dataflow, self.cycles,
                   self.start))


def default_thread_order(num_threads):
    """Identity placement. Swappable: see the module docstring."""
    return list(range(num_threads))


def list_schedule(tiles, tile_cycles, num_threads, dataflow,
                  switch_penalty=0, previous_dataflow=None,
                  thread_order=None):
    """Assign tiles to threads, earliest-free first, and stamp start/end.

    `tile_cycles` maps a tile to its compute-only cycle count. `switch_penalty`
    is charged on a thread whose previous tile ran a different dataflow.
    Returns (jobs, makespan) with makespan measured without contention.
    """
    assert num_threads >= 1
    order = list(thread_order) if thread_order is not None \
        else default_thread_order(num_threads)
    assert sorted(order) == list(range(num_threads)), 'thread order must be a permutation'

    free_at = [0] * num_threads
    last_dataflow = [previous_dataflow] * num_threads
    jobs = []

    for tile in tiles:
        slot = min(range(num_threads), key=lambda index: (free_at[index], index))
        thread_id = order[slot]

        job = tile_job(tile, thread_id, dataflow, tile_cycles(tile))
        if last_dataflow[slot] is not None and last_dataflow[slot] != dataflow:
            job.switch_penalty = switch_penalty

        job.start = free_at[slot] + job.switch_penalty
        job.end = job.start + job.cycles
        free_at[slot] = job.end
        last_dataflow[slot] = dataflow
        jobs.append(job)

    return jobs, max(free_at) if jobs else 0


def waves(jobs, num_threads):
    """Group jobs into barrier-separated waves of at most num_threads tiles.

    Jobs keep the order list_schedule produced, so a wave is what actually runs
    concurrently when every tile costs the same.
    """
    return [jobs[start:start + num_threads]
            for start in range(0, len(jobs), num_threads)]


def wave_makespan(jobs, num_threads):
    """Compute-only makespan under the wave model, for comparison.

    Each wave costs its longest tile. Equal to the list-scheduling makespan
    when the tiles are equal, and never smaller.
    """
    total = 0
    for wave in waves(jobs, num_threads):
        total += max(job.cycles + job.switch_penalty for job in wave)
    return total


def balanced_makespan_bound(tile_cycle_list, num_threads):
    """The Phase 2 objective: max(longest tile, total / threads).

    Kept here so the arbiter can report how much of the difference between
    Phase 2 and Phase 3 came from scheduling rather than from contention.
    """
    if not tile_cycle_list:
        return 0
    return max(max(tile_cycle_list),
               math.ceil(sum(tile_cycle_list) / num_threads))
