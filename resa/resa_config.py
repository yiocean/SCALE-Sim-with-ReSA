"""Reader for configs/resa.cfg.

The file is both a valid SCALE-Sim config and the ReSA cost model: SCALE-Sim
consumes the standard sections, this module consumes the [resa_*] ones. Every
cost-model number lives in the file, never in code (CLAUDE.md rule 4), so that
the sensitivity of a conclusion to any estimate can be measured by editing the
config alone.
"""

import configparser
import copy
import os

from resa.buffer_groups import build_group_map
from scalesim.scale_config import scale_config


class resa_config:
    """ReSA hardware parameters plus the SCALE-Sim config they sit alongside."""

    def __init__(self):
        # [resa_array]
        self.physical_array_rows = 128
        self.physical_array_cols = 128
        self.sub_array_rows = 32
        self.sub_array_cols = 32
        self.grid_rows = 4
        self.grid_cols = 4
        self.clock_frequency_mhz = 700
        self.operand_bits = 16

        # [resa_buffer]
        self.sub_arrays_per_group = 4
        self.num_buffer_groups = 4
        self.ports_per_group = 1
        self.crossbar_ports = 4
        self.buffer_kb_per_group = 1536
        self.onchip_bandwidth = 1024
        self.port_model = 'bandwidth'
        self.arbitration_policy = 'round_robin'
        self.multicast_scope = 'same_cycle_same_group_same_address'
        self.group_geometry = 'quadrant'
        self.custom_group_map = ()
        self.group_of_sub_array = ()

        # [resa_noc]
        self.noc_hop_latency_cycles = 1
        self.noc_unidirectional = True

        # [resa_cost]
        self.dataflow_switch_penalty = 'auto'

        # [resa_memory]
        self.offchip_bandwidth = 179
        self.per_group_bandwidth = 44
        self.bandwidth_enforcement = 'arbiter'
        self.bandwidth_remainder_policy = 'unassigned'

        # [resa_search]
        self.ranking_tilings_per_cell = 40
        self.ranking_margin_percent = 25.0

        self.config_file = ''
        self.scale_cfg = scale_config()
        self.valid_conf_flag = False

    #
    def read_conf_file(self, conf_file_in):
        """Read a resa.cfg. The same file is handed to scale_config as-is."""
        if not os.path.exists(conf_file_in):
            raise FileNotFoundError('resa config not found: ' + conf_file_in)

        parser = configparser.ConfigParser()
        parser.read(conf_file_in)

        section = 'resa_array'
        self.physical_array_rows = parser.getint(section, 'PhysicalArrayHeight')
        self.physical_array_cols = parser.getint(section, 'PhysicalArrayWidth')
        self.sub_array_rows = parser.getint(section, 'SubArrayHeight')
        self.sub_array_cols = parser.getint(section, 'SubArrayWidth')
        self.grid_rows = parser.getint(section, 'SubArrayGridRows')
        self.grid_cols = parser.getint(section, 'SubArrayGridCols')
        self.clock_frequency_mhz = parser.getint(section, 'ClockFrequencyMHz')
        self.operand_bits = parser.getint(section, 'OperandBits')

        section = 'resa_buffer'
        self.sub_arrays_per_group = parser.getint(section, 'SubArraysPerGroup')
        self.num_buffer_groups = parser.getint(section, 'NumBufferGroups')
        self.ports_per_group = parser.getint(section, 'ReadWritePortsPerGroup')
        self.crossbar_ports = parser.getint(section, 'CrossbarPorts')
        self.buffer_kb_per_group = parser.getint(section, 'BufferSizeKbPerGroup')
        self.onchip_bandwidth = parser.getint(section, 'OnChipBandwidthWordsPerCycle')
        self.port_model = parser.get(section, 'PortModel', fallback='bandwidth').strip()
        self.arbitration_policy = parser.get(section, 'ArbitrationPolicy').strip()
        self.multicast_scope = parser.get(section, 'MulticastScope').strip()
        self.group_geometry = parser.get(section, 'GroupGeometry').strip()
        raw_map = parser.get(section, 'CustomGroupMap', fallback='').strip()
        self.custom_group_map = tuple(int(value) for value in raw_map.split(',')
                                      if value.strip()) if raw_map else ()

        section = 'resa_noc'
        self.noc_hop_latency_cycles = parser.getint(section, 'HopLatencyCycles')
        self.noc_unidirectional = parser.getboolean(section, 'Unidirectional')

        section = 'resa_cost'
        self.dataflow_switch_penalty = \
            parser.get(section, 'DataflowSwitchPenaltyCycles').strip()

        section = 'resa_memory'
        self.offchip_bandwidth = parser.getint(section, 'OffChipBandwidthWordsPerCycle')
        self.per_group_bandwidth = parser.getint(section, 'PerGroupBandwidthWordsPerCycle')
        self.bandwidth_enforcement = parser.get(section, 'BandwidthEnforcement').strip()
        self.bandwidth_remainder_policy = \
            parser.get(section, 'BandwidthRemainderPolicy').strip()

        section = 'resa_search'
        self.ranking_tilings_per_cell = parser.getint(section, 'RankingTilingsPerCell')
        self.ranking_margin_percent = parser.getfloat(section, 'RankingMarginPercent')

        self.scale_cfg = scale_config()
        self.scale_cfg.read_conf_file(conf_file_in)

        self.config_file = conf_file_in
        self.valid_conf_flag = True
        self._check()

    #
    def _check(self):
        """Consistency checks that would otherwise surface as silent wrong numbers."""
        assert self.sub_array_rows * self.grid_rows == self.physical_array_rows, \
            'sub-array grid does not tile the physical array along rows'
        assert self.sub_array_cols * self.grid_cols == self.physical_array_cols, \
            'sub-array grid does not tile the physical array along cols'

        num_sub_arrays = self.grid_rows * self.grid_cols
        assert num_sub_arrays == self.num_buffer_groups * self.sub_arrays_per_group, \
            'buffer groups do not account for every sub-array'
        assert self.crossbar_ports == self.sub_arrays_per_group, \
            'the data router crossbar must reach every sub-array of its group'

        self.group_of_sub_array = build_group_map(
            self.group_geometry, self.grid_rows, self.grid_cols,
            self.sub_arrays_per_group,
            custom=self.custom_group_map or None)

        # docs/resa_spec.md section 3: ReSA drops the upward and right-to-left
        # links. A bidirectional NoC would be Planaria, not ReSA.
        assert self.noc_unidirectional, \
            'ReSA NoC is unidirectional (spec section 3); set Unidirectional: True'

        assert self.noc_hop_latency_cycles >= 0
        assert self.ports_per_group >= 1
        assert self.offchip_bandwidth > 0 and self.per_group_bandwidth > 0

        # The whole-chip budget must not be smaller than what the groups are
        # allowed to draw, otherwise the arbiter's per-group cap is meaningless.
        assert self.per_group_bandwidth * self.num_buffer_groups <= self.offchip_bandwidth, \
            'per-group bandwidths oversubscribe the off-chip budget'

        if self.dataflow_switch_penalty != 'auto':
            assert int(self.dataflow_switch_penalty) >= 0

        assert self.port_model in ('bandwidth', 'subarray_serial', 'both'), \
            'unknown PortModel: ' + self.port_model
        assert self.onchip_bandwidth > 0
        assert self.onchip_bandwidth >= self.num_buffer_groups, \
            'on-chip bandwidth cannot give every group at least one word/cycle'

        assert self.ranking_tilings_per_cell >= 1, \
            'the ranking must keep at least the best tiling per cell'
        assert self.ranking_margin_percent >= 0

        assert self.get_bandwidth_remainder() >= 0
        policy = self.bandwidth_remainder_policy
        if policy.startswith('group:'):
            group_id = int(policy.split(':', 1)[1])
            assert 0 <= group_id < self.num_buffer_groups, \
                'BandwidthRemainderPolicy names a group that does not exist'
        else:
            assert policy in ('unassigned', 'round_robin'), \
                'unknown BandwidthRemainderPolicy: ' + policy

    # ---------------------------------------------------------------- getters
    def get_physical_array_dims(self):
        """(rows, cols) of the whole PE array."""
        return self.physical_array_rows, self.physical_array_cols

    def get_sub_array_dims(self):
        """(rows, cols) of one sub-array, i.e. the 1x1 fission."""
        return self.sub_array_rows, self.sub_array_cols

    def get_sub_array_grid(self):
        """(rows, cols) of the sub-array grid, 4x4 in the paper."""
        return self.grid_rows, self.grid_cols

    def get_num_sub_arrays(self):
        return self.grid_rows * self.grid_cols

    def get_buffer_group_layout(self):
        """(num groups, sub-arrays per group, R/W ports per group)."""
        return self.num_buffer_groups, self.sub_arrays_per_group, self.ports_per_group

    def get_group_port_budget(self):
        """Words one buffer group's single R/W port can move per cycle.

        The whole-chip on-chip bandwidth split evenly over the groups, floored
        so the groups together never exceed it. Reads and writes share it --
        that is what "single read-and-write port" costs in this model.
        """
        return self.onchip_bandwidth // self.num_buffer_groups

    def get_onchip_bandwidth(self):
        return self.onchip_bandwidth

    def get_port_model(self):
        """'bandwidth', 'subarray_serial' or 'both' (section 8, item 13)."""
        return self.port_model

    def set_port_model(self, port_model):
        """Swap the port model in memory, for the Phase 3 sensitivity sweep."""
        assert port_model in ('bandwidth', 'subarray_serial', 'both')
        self.port_model = port_model
        return self.port_model

    def get_buffer_kb_per_group(self):
        """Total SCALE-Sim kB behind one buffer group, all operand types."""
        return self.buffer_kb_per_group

    def get_arbitration_policy(self):
        return self.arbitration_policy

    def get_multicast_scope(self):
        return self.multicast_scope

    def get_group_map(self):
        """Flat sub-array id (row-major over the grid) -> buffer group id.

        The arbiter reads group membership from here and never assumes a
        geometry; see resa/buffer_groups.py for why both presets exist.
        """
        assert self.valid_conf_flag, 'resa config not read yet'
        return self.group_of_sub_array

    def get_group_geometry(self):
        return self.group_geometry

    def set_group_geometry(self, geometry, custom=None):
        """Swap the geometry in memory, for the Phase 3 sensitivity sweep."""
        self.group_geometry = geometry
        if custom is not None:
            self.custom_group_map = tuple(int(value) for value in custom)
        self.group_of_sub_array = build_group_map(
            geometry, self.grid_rows, self.grid_cols, self.sub_arrays_per_group,
            custom=self.custom_group_map or None)
        return self.group_of_sub_array

    def get_noc_params(self):
        """(hop latency in cycles, unidirectional flag)."""
        return self.noc_hop_latency_cycles, self.noc_unidirectional

    def get_dataflow_switch_penalty(self, arr_row, arr_col):
        """Reconfiguration cost in cycles (docs/resa_spec.md section 8, item 2).

        'auto' is arr_row + arr_col, i.e. the time to drain the pipeline of the
        logical array being reconfigured.
        """
        if self.dataflow_switch_penalty == 'auto':
            return arr_row + arr_col
        return int(self.dataflow_switch_penalty)

    def get_bandwidth_params(self):
        """(whole-chip words/cycle, per-group words/cycle, who enforces it)."""
        return (self.offchip_bandwidth, self.per_group_bandwidth,
                self.bandwidth_enforcement)

    def get_bandwidth_remainder(self):
        """Words/cycle the per-group floor leaves unallocated (3 in the paper's numbers)."""
        return (self.offchip_bandwidth
                - self.per_group_bandwidth * self.num_buffer_groups)

    def get_ranking_limits(self):
        """(tilings kept per ranking cell, margin percent).

        Phase 3 re-ranks inside what these keep, so they bound how much of
        Algorithm 1's space survives the compute-only pruning.
        """
        return self.ranking_tilings_per_cell, self.ranking_margin_percent

    def get_bandwidth_remainder_policy(self):
        """'unassigned', 'round_robin', or 'group:<id>'. Phase 3 acts on it."""
        return self.bandwidth_remainder_policy

    def get_clock_frequency_hz(self):
        return self.clock_frequency_mhz * 1_000_000

    def get_operand_bits(self):
        return self.operand_bits

    #
    def get_scale_config(self):
        """A fresh copy of the SCALE-Sim config carried by this file.

        A copy, because callers inject per-fission array dimensions into it and
        must not disturb the shared one. Note that scale_config.set_offsets()
        is buggy upstream (it assigns ifmap_offset twice and never sets
        ofmap_offset), so offsets are only ever taken from the config file.
        """
        assert self.valid_conf_flag, 'resa config not read yet'
        return copy.deepcopy(self.scale_cfg)
