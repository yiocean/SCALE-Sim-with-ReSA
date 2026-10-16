"""Command line entry point: python -m resa <command>

Commands:

    run         search + evaluate in one go -- the usual entry point
    search      Algorithm 1 over every layer -> profiling table + search ranking
    evaluate    Algorithm 2 on top -> per-layer report + per-model summary
    rerank      Phase 3 analysis: which fission wins under contention alone
    baseline    native SCALE-Sim on the monolithic 128x128 array, for reference

Flags mirror scalesim.scale: -c config, -t topology, -p output path. Without
-t the four workloads named in docs/resa_plan.md are used.

    python -m resa -c configs/resa.cfg -t topologies/conv_nets/alexnet.csv \
                   -p out/alexnet run

`evaluate` and `rerank` both read the ranking `search` writes, so run `search`
first (or just use `run`).
"""

import argparse
import collections
import csv
import math
import os
import sys
import time

from resa import cycle_model, memory_policy, reports
from resa.fission import fission_space
from resa.resa_config import resa_config
from resa.resa_sim import evaluate_layer, layer_operands
from resa.tensor_split import layer_tiling
from resa.tile_search import (search_model, tile_shapes,
                              write_profiling_table, write_ranking_table)
from scalesim.topology_utils import topologies

WORKLOADS = {
    'MobileNet': 'topologies/conv_nets/mobilenet.csv',
    'GoogleNet': 'topologies/conv_nets/Googlenet.csv',
    'ResNet50': 'topologies/conv_nets/Resnet50_resa.csv',
    'UNet_maestro': 'topologies/conv_nets/UNet_maestro_resa.csv',
}

# Cycle-accurate contention on UNet_maestro is too slow at a useful shortlist
# depth. Under the mainline port model fast and accurate agree exactly, so the
# fast path is used there and the report records which mode each layer used.
#
# This is a DENYLIST on purpose. It used to be an allowlist of the three fast
# workloads, which silently downgraded every topology passed with -t -- a model
# the list had never heard of got the fast path without saying so. Anything not
# named here runs cycle-accurate; --force-accurate overrides even this.
SLOW_MODELS = ('UNet_maestro',)

RANKING_CSV = 'RESA_SEARCH_RANKING_compute_only.csv'
PROFILING_CSV = 'RESA_PROFILING_TABLE_compute_only.csv'
PHASE3_CSV = 'RESA_PROFILING_TABLE_phase3.csv'
RERANK_CSV = 'RESA_RERANK_phase3.csv'
COMPUTE_CSV = 'RESA_COMPUTE_REPORT.csv'
SUMMARY_CSV = 'RESA_SUMMARY.csv'


def _load_topology(path, input_type='conv'):
    """Load a topology the way scalesim.scale does, conv or gemm.

    GEMM files are M, N, K per row rather than a convolution description;
    topology_utils lowers them to the same operand matrices either way, so
    nothing downstream of here needs to know which it was.
    """
    topo_obj = topologies()
    if input_type == 'gemm':
        topo_obj.load_arrays_gemm(topofile=path)
    else:
        topo_obj.load_arrays(topofile=path)
    return topo_obj


def _workloads(args):
    """{name: topology path} for this run.

    -t / --topology points at any topology csv, the way scalesim.scale does; the
    file's stem becomes the model name. Without it the four workloads named in
    docs/resa_plan.md are used, optionally narrowed by --models.
    """
    if getattr(args, 'topology', None):
        path = args.topology
        if not os.path.exists(path):
            raise SystemExit('topology file not found: ' + path)
        return {os.path.splitext(os.path.basename(path))[0]: path}

    names = args.models
    if not names:
        return dict(WORKLOADS)
    for name in names:
        if name not in WORKLOADS:
            raise SystemExit('unknown model %r; choose from %s, or point -t at '
                             'any topology csv' % (name, ', '.join(WORKLOADS)))
    return {name: WORKLOADS[name] for name in names}


def _config(args):
    cfg = resa_config()
    cfg.read_conf_file(args.config)
    if getattr(args, 'port_model', None):
        cfg.set_port_model(args.port_model)
    if getattr(args, 'geometry', None):
        cfg.set_group_geometry(args.geometry)
    return cfg


def _read_ranking(out_dir):
    path = os.path.join(out_dir, RANKING_CSV)
    if not os.path.exists(path):
        raise SystemExit('%s not found -- run `python -m resa search` first' % path)
    with open(path) as handle:
        rows = list(csv.DictReader(line for line in handle
                                   if not line.startswith('#')))
    by_layer = collections.defaultdict(list)
    for row in rows:
        by_layer[(row['model'], int(row['layer_id']))].append(row)
    return by_layer


# --------------------------------------------------------------------------

def cmd_search(args):
    """Algorithm 1: search every layer, write the profiling table and ranking."""
    cfg = _config(args)
    entries, rankings = [], {}

    for model, topology in _workloads(args).items():
        topo_obj = _load_topology(topology, getattr(args, "input_type", "conv"))
        started = time.time()
        examined = [0]

        def progress(entry, count, seen=examined):
            seen[0] += count

        model_entries, model_rankings = search_model(
            cfg, topo_obj, model, verify=not args.no_verify,
            progress=progress, return_ranking=True)

        entries += model_entries
        for layer_id, ranking in model_rankings.items():
            rankings[(model, layer_id)] = ranking

        fissions = collections.Counter('%dx%d' % (e.R, e.S) for e in model_entries)
        dataflows = collections.Counter(e.dataflow for e in model_entries)
        truncated = sum(1 for r in model_rankings.values()
                        for cell in r.values() if cell.truncated)
        print('%-14s %3d layers  %8d candidates  %6.1fs'
              % (model, len(model_entries), examined[0], time.time() - started))
        print('   fission  %s' % dict(fissions))
        print('   dataflow %s' % dict(dataflows))
        print('   ranking  %d cells truncated' % truncated)
        sys.stdout.flush()

    write_profiling_table(entries, os.path.join(args.out, PROFILING_CSV))
    write_ranking_table(entries, rankings, os.path.join(args.out, RANKING_CSV))
    print('\nwrote %s and %s' % (PROFILING_CSV, RANKING_CSV))


def cmd_rerank(args):
    """Phase 3: re-rank the search output with buffer-port contention on."""
    cfg = _config(args)
    by_layer = _read_ranking(args.out)
    space = fission_space(cfg)
    rows = []

    for model, topology in _workloads(args).items():
        topo_obj = _load_topology(topology, getattr(args, "input_type", "conv"))
        started = time.time()
        picks = collections.Counter()
        boundary = 0
        accurate = (args.mode == 'accurate'
                    and (model not in SLOW_MODELS or args.force_accurate))

        for layer_id in range(topo_obj.get_num_layers()):
            candidates = by_layer[(model, layer_id)]
            if not candidates:
                continue
            operands = layer_operands(cfg, topo_obj, layer_id)

            scored = []
            for row in candidates:
                result = evaluate_layer(
                    cfg, operands, space.get(int(row['R']), int(row['S'])),
                    row['dataflow'], int(row['m_tile_size']),
                    int(row['n_tile_size']), mode='fast')
                scored.append((result.total_cycles, result.compute_makespan, row))
            scored.sort(key=lambda item: (item[0], int(item[2]['rank_in_cell'])))

            best = scored[0]
            mode = 'fast'
            if accurate:
                mode = 'accurate'
                rescored = []
                for _, _, row in scored[:args.shortlist]:
                    result = evaluate_layer(
                        cfg, operands, space.get(int(row['R']), int(row['S'])),
                        row['dataflow'], int(row['m_tile_size']),
                        int(row['n_tile_size']), mode='accurate')
                    rescored.append((result.total_cycles,
                                     result.compute_makespan, row))
                rescored.sort(key=lambda item: (item[0],
                                                int(item[2]['rank_in_cell'])))
                best = rescored[0]

            total, compute, row = best
            picks['%sx%s' % (row['R'], row['S'])] += 1
            peers = [int(other['rank_in_cell']) for other in candidates
                     if (other['R'], other['S'], other['dataflow'])
                     == (row['R'], row['S'], row['dataflow'])]
            if row['cell_truncated'] == 'True' and int(row['rank_in_cell']) == max(peers):
                boundary += 1

            rows.append(dict(
                model=model, layer_id=layer_id, layer_name=row['layer_name'],
                R=row['R'], S=row['S'], dataflow=row['dataflow'],
                m_tile_size=row['m_tile_size'], n_tile_size=row['n_tile_size'],
                tiles=row['tiles'], threads=row['threads'],
                compute_cycles=compute, total_cycles=total, mode=mode,
                port_model=cfg.get_port_model(),
                geometry=cfg.get_group_geometry(),
                rank_in_cell=row['rank_in_cell'],
                cell_truncated=row['cell_truncated']))

        print('%-14s %3d layers  %6.1fs  fission %s'
              % (model, sum(1 for r in rows if r['model'] == model),
                 time.time() - started, dict(picks)))
        if boundary:
            print('   WARNING %d winner(s) sat on a truncated cell boundary -- '
                  'raise [resa_search] RankingTilingsPerCell and re-run search'
                  % boundary)
        sys.stdout.flush()

    name = PHASE3_CSV if (cfg.get_port_model() == 'bandwidth'
                          and cfg.get_group_geometry() == 'quadrant') else RERANK_CSV
    path = os.path.join(args.out, name)
    os.makedirs(args.out, exist_ok=True)
    with open(path, 'w', newline='') as handle:
        handle.write('# Phase 3 re-ranking -- based on our reimplementation of ReSA.\n'
                     '# PortModel=%s, GroupGeometry=%s. See docs/resa_spec.md section 8.\n'
                     % (cfg.get_port_model(), cfg.get_group_geometry()))
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('\nwrote %s (%d rows)' % (name, len(rows)))


def cmd_evaluate(args):
    """Phase 4: Algorithm 2 on top of the chosen configurations."""
    cfg = _config(args)
    by_layer = _read_ranking(args.out)
    space = fission_space(cfg)
    monolithic = space.monolithic()
    bandwidth = cfg.get_bandwidth_params()[0]
    clock = cfg.get_clock_frequency_hz()
    array_rows, array_cols = cfg.get_physical_array_dims()
    num_pes = array_rows * array_cols

    all_rows, summaries = [], []
    for model, topology in _workloads(args).items():
        topo_obj = _load_topology(topology, getattr(args, "input_type", "conv"))
        started = time.time()
        rows, picks = [], collections.Counter()

        for layer_id in range(topo_obj.get_num_layers()):
            operands = layer_operands(cfg, topo_obj, layer_id)
            footprint = memory_policy.ifmap_footprint(topo_obj, layer_id)
            macs = operands.m * operands.n * operands.k

            base = evaluate_layer(cfg, operands, monolithic, 'ws',
                                  operands.m, operands.n, mode='fast')
            base_traffic = memory_policy.layer_traffic(
                operands, layer_tiling(operands.m, operands.n), footprint=footprint)
            base_wait = memory_policy.resa_wait(base_traffic, bandwidth,
                                                base.total_cycles)
            base_total = base.total_cycles + base_wait.wait_cycles

            best = None
            for row in by_layer[(model, layer_id)]:
                result = evaluate_layer(
                    cfg, operands, space.get(int(row['R']), int(row['S'])),
                    row['dataflow'], int(row['m_tile_size']),
                    int(row['n_tile_size']), mode='fast')
                compute = result.total_cycles
                tiling = layer_tiling(operands.m, operands.n,
                                      m_tile_size=int(row['m_tile_size']),
                                      n_tile_size=int(row['n_tile_size']))
                traffic = memory_policy.layer_traffic(operands, tiling,
                                                      footprint=footprint)
                wait = memory_policy.resa_wait(traffic, bandwidth, compute)
                total = compute + wait.wait_cycles
                if best is None or total < best[0]:
                    best = (total, compute, wait, traffic, row, result, tiling)

            total, compute, wait, traffic, row, result, tiling = best
            fission_obj = space.get(int(row['R']), int(row['S']))
            tile_rows, tile_cols = fission_obj.logical_dims
            mapping_eff, compute_util = cycle_model.weighted_utilisation(
                row['dataflow'],
                tile_shapes(tiling.num_operand_rows, tiling.num_filters,
                            tiling.m_tile_size, tiling.n_tile_size),
                operands.k, tile_rows, tile_cols)
            base_mapping, base_util = cycle_model.weighted_utilisation(
                'ws', [(operands.m, operands.n, 1)], operands.k,
                *monolithic.logical_dims)
            planaria = memory_policy.planaria_wait(traffic, bandwidth, compute)
            picks['%sx%s' % (row['R'], row['S'])] += 1

            rows.append(dict(
                model=model, layer_id=layer_id, layer_name=row['layer_name'],
                R=int(row['R']), S=int(row['S']), dataflow=row['dataflow'],
                m_tile_size=int(row['m_tile_size']),
                n_tile_size=int(row['n_tile_size']),
                tiles=traffic.num_tiles, threads=int(row['threads']),
                compute_cycles=compute,
                stall_cycles=result.contention.stall_cycles,
                memory_wait_cycles=wait.wait_cycles,
                total_cycles=total, dram_words=traffic.total_words,
                pe_utilisation_pct=100.0 * macs / (total * num_pes),
                mapping_efficiency_pct=100.0 * mapping_eff,
                compute_util_pct=100.0 * compute_util,
                baseline_compute_cycles=base.total_cycles,
                baseline_stall_cycles=base.contention.stall_cycles,
                baseline_memory_wait_cycles=base_wait.wait_cycles,
                baseline_total_cycles=base_total,
                baseline_pe_utilisation_pct=100.0 * macs / (base_total * num_pes),
                baseline_mapping_efficiency_pct=100.0 * base_mapping,
                baseline_compute_util_pct=100.0 * base_util,
                speedup=base_total / total,
                planaria_policy_wait_cycles=planaria.wait_cycles,
                memory_wait_ratio_planaria_over_resa=(
                    planaria.wait_cycles / wait.wait_cycles
                    if wait.wait_cycles else 0.0)))

        summary = reports.summarise(model, rows, clock)
        summaries.append(summary)
        all_rows += rows
        print('%-14s %3d layers %6.1fs  speedup %.2fx  throughput %.1f inf/s'
              % (model, len(rows), time.time() - started,
                 summary['speedup'], summary['throughput_inf_per_s']))
        print('   fission %s' % dict(picks))
        print('   memory wait %.1f%% of total   Planaria/ReSA wait %.2fx'
              % (summary['memory_wait_fraction_pct'],
                 summary['memory_wait_ratio_planaria_over_resa']))
        sys.stdout.flush()

    reports.write_compute_report(all_rows, os.path.join(args.out, COMPUTE_CSV))
    reports.write_summary(summaries, os.path.join(args.out, SUMMARY_CSV), clock)
    geomean = 1.0
    for summary in summaries:
        geomean *= summary['speedup']
    print('\ngeomean speedup vs monolithic %dx%d WS baseline: %.2fx'
          % (array_rows, array_cols, geomean ** (1.0 / len(summaries))))
    print('wrote %s and %s' % (COMPUTE_CSV, SUMMARY_CSV))


def cmd_run(args):
    """search + evaluate in one go, the closest thing to `scalesim.scale`.

    `rerank` is deliberately not part of this: evaluate reads the ranking that
    search writes and does its own contention scoring, so rerank is a separate
    Phase 3 analysis (which fission wins under contention, and how sensitive
    that is to the two undecided details), not a step on the way to the report.
    """
    cmd_search(args)
    print()
    cmd_evaluate(args)


def cmd_baseline(args):
    """Native SCALE-Sim on the monolithic array, straight from scalesim/.

    Reports land in <out>/baseline/<model>/<run_name>/, where run_name comes
    from the baseline config. SCALE-Sim creates the leaf with os.mkdir rather
    than makedirs, so the parent has to exist first.
    """
    from scalesim.scale_sim import scalesim

    for model, topology in _workloads(args).items():
        print('=== %s' % model)
        destination = os.path.join(args.out, 'baseline', model)
        os.makedirs(destination, exist_ok=True)
        sim = scalesim(save_disk_space=True, verbose=True,
                       config=args.baseline_config,
                       topology=topology,
                       layout=args.layout, input_type_gemm=False)
        sim.run_scale(top_path=destination)


# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog='python -m resa', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Short flags mirror scalesim.scale so the muscle memory carries over:
    #   -c config   -t topology   -p output path
    parser.add_argument('-c', '--config', default='configs/resa.cfg',
                        help='ReSA config (default: configs/resa.cfg)')
    parser.add_argument('-t', '--topology', default=None,
                        help='any topology csv; without it the four workloads '
                             'in docs/resa_plan.md are used')
    parser.add_argument('-i', '--input-type', choices=('conv', 'gemm'),
                        default='conv', dest='input_type',
                        help='topology format, mirroring scalesim.scale -i')
    parser.add_argument('-p', '--out', default='out/resa',
                        help='output directory (default: out/resa)')
    parser.add_argument('-m', '--models', nargs='*', default=None,
                        help='subset of %s' % ', '.join(WORKLOADS))
    sub = parser.add_subparsers(dest='command', required=True)

    search = sub.add_parser('search', help='Algorithm 1 tile/dataflow/fission search')
    search.add_argument('--no-verify', action='store_true',
                        help='skip re-running each winner through SCALE-Sim')
    search.set_defaults(func=cmd_search)

    rerank = sub.add_parser('rerank', help='re-rank with buffer-port contention')
    rerank.add_argument('--mode', choices=('fast', 'accurate'), default='accurate')
    rerank.add_argument('--shortlist', type=int, default=8)
    rerank.add_argument('--port-model', choices=('bandwidth', 'subarray_serial', 'both'))
    rerank.add_argument('--geometry', choices=('quadrant', 'row'))
    rerank.add_argument('--force-accurate', action='store_true',
                        help='use the accurate mode even on UNet_maestro (slow)')
    rerank.set_defaults(func=cmd_rerank)

    evaluate = sub.add_parser('evaluate', help='Algorithm 2 + reports')
    evaluate.add_argument('--port-model', choices=('bandwidth', 'subarray_serial', 'both'))
    evaluate.add_argument('--geometry', choices=('quadrant', 'row'))
    evaluate.set_defaults(func=cmd_evaluate)

    run = sub.add_parser('run', help='search + evaluate in one go')
    run.add_argument('--no-verify', action='store_true',
                     help='skip re-running each winner through SCALE-Sim')
    run.add_argument('--port-model', choices=('bandwidth', 'subarray_serial', 'both'))
    run.add_argument('--geometry', choices=('quadrant', 'row'))
    run.set_defaults(func=cmd_run)

    baseline = sub.add_parser('baseline', help='native SCALE-Sim reference run')
    baseline.add_argument('--baseline-config', default='configs/resa_baseline.cfg')
    baseline.add_argument('--layout', default='layouts/conv_nets/test.csv')
    baseline.set_defaults(func=cmd_baseline)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    args.func(args)


if __name__ == '__main__':
    main()
