"""Phase 4 reports: per-layer detail and per-model summary.

Every file carries the same provenance header. These numbers are based on our
reimplementation of ReSA, not on the paper, and the modelling decisions that
shape them are listed in docs/resa_spec.md section 8.
"""

import csv
import os

HEADER = (
    '# Based on our reimplementation of ReSA (Lee & Yeh, ACM TACO 21(3), 2024).\n'
    '# Numbers are NOT claimed to match the paper. The modelling decisions that\n'
    '# deviate from it are listed in docs/resa_spec.md section 8; the ones that\n'
    '# shape this report are:\n'
    '#   item 1  K splitting disabled, so no cross-sub-array reduction\n'
    '#   item 9  M tiles are sized, not shaped: no 2D (oh x ow) tile\n'
    '#   item 10 the search objective is a makespan bound, not a real schedule\n'
    '#   item 12 "single R/W port" read as a per-group word budget (mainline)\n'
    '#   item 13 off-chip bandwidth 2864 bits/cycle, per Table 3, not the text\n'
    '#   item 14 topology_utils rounds ofmap up, so some windows fall outside\n'
    '# Baseline is a monolithic 128x128 weight-stationary array (spec section 9)\n'
    '# running under the SAME off-chip bandwidth and the same memory policy, so\n'
    '# the comparison is like for like.\n')

COMPUTE_FIELDS = [
    'model', 'layer_id', 'layer_name', 'R', 'S', 'dataflow',
    'm_tile_size', 'n_tile_size', 'tiles', 'threads',
    'compute_cycles', 'memory_wait_cycles', 'total_cycles',
    'dram_words', 'pe_utilisation_pct',
    'baseline_compute_cycles', 'baseline_memory_wait_cycles',
    'baseline_total_cycles', 'baseline_pe_utilisation_pct', 'speedup',
    'planaria_policy_wait_cycles', 'memory_wait_ratio_planaria_over_resa',
]

SUMMARY_FIELDS = [
    'model', 'layers', 'total_cycles', 'baseline_total_cycles', 'speedup',
    'throughput_inf_per_s', 'baseline_throughput_inf_per_s',
    'compute_cycles', 'memory_wait_cycles', 'memory_wait_fraction_pct',
    'dram_words', 'mean_pe_utilisation_pct', 'baseline_mean_pe_utilisation_pct',
    'planaria_policy_wait_cycles', 'memory_wait_ratio_planaria_over_resa',
]


def _write(path, fields, rows, extra_header=''):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        handle.write(HEADER)
        if extra_header:
            handle.write(extra_header)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return path


def write_compute_report(rows, path):
    """One row per layer."""
    return _write(path, COMPUTE_FIELDS, rows)


def write_summary(rows, path, clock_hz):
    """One row per model."""
    extra = ('# Throughput is %d / total_cycles at batch size 1 (paper section 6).\n'
             % clock_hz)
    return _write(path, SUMMARY_FIELDS, rows, extra_header=extra)


def summarise(model, layer_rows, clock_hz):
    """Fold a model's layer rows into one summary row."""
    total = sum(row['total_cycles'] for row in layer_rows)
    baseline_total = sum(row['baseline_total_cycles'] for row in layer_rows)
    compute = sum(row['compute_cycles'] for row in layer_rows)
    wait = sum(row['memory_wait_cycles'] for row in layer_rows)
    planaria = sum(row['planaria_policy_wait_cycles'] for row in layer_rows)

    return {
        'model': model,
        'layers': len(layer_rows),
        'total_cycles': total,
        'baseline_total_cycles': baseline_total,
        'speedup': baseline_total / total if total else 0.0,
        'throughput_inf_per_s': clock_hz / total if total else 0.0,
        'baseline_throughput_inf_per_s': clock_hz / baseline_total if baseline_total else 0.0,
        'compute_cycles': compute,
        'memory_wait_cycles': wait,
        'memory_wait_fraction_pct': 100.0 * wait / total if total else 0.0,
        'dram_words': sum(row['dram_words'] for row in layer_rows),
        'mean_pe_utilisation_pct':
            sum(row['pe_utilisation_pct'] for row in layer_rows) / len(layer_rows),
        'baseline_mean_pe_utilisation_pct':
            sum(row['baseline_pe_utilisation_pct'] for row in layer_rows) / len(layer_rows),
        'planaria_policy_wait_cycles': planaria,
        'memory_wait_ratio_planaria_over_resa': planaria / wait if wait else 0.0,
    }
