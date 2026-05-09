"""Compatibility wrapper for AGV layout builders.

New code should import from rware.agv_layouts. This module keeps earlier
scripts working while the AGV extension is split into clearer components.
"""

from rware.agv_layouts import AGVLayout, build_four_way_intersection_layout, build_graph_rware_intersection_v1

__all__ = [
    "AGVLayout",
    "build_four_way_intersection_layout",
    "build_graph_rware_intersection_v1",
]
