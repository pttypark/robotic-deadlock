"""Render the FCFS double-cross experiment layout preview."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import build_fcfs_double_cross_shared_area_layout


CELL = 28
MARGIN = 14

COLORS = {
    "storage": (216, 219, 212),
    "road": (250, 250, 246),
    "grid": (150, 154, 148),
    "conflict": (226, 164, 105),
    "waiting": (220, 170, 63),
    "start": (137, 205, 82),
    "exit": (114, 114, 238),
    "text": (92, 70, 28),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render FCFS double-cross layout preview.")
    parser.add_argument(
        "--output",
        default=str(Path("outputs") / "layout_previews" / "fcfs_double_cross_shared_area_v2.png"),
    )
    parser.add_argument("--corridor-length", type=int, default=8)
    args = parser.parse_args()

    layout = build_fcfs_double_cross_shared_area_layout(corridor_length=args.corridor_length)
    frame = draw_layout(layout)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), frame)
    print(output)


def draw_layout(layout) -> np.ndarray:
    rows, cols = layout.grid_size
    height = rows * CELL + 2 * MARGIN
    width = cols * CELL + 2 * MARGIN
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    road_positions = {node.position for node in layout.graph.nodes.values()}

    for row in range(rows):
        for col in range(cols):
            color = COLORS["road"] if (row, col) in road_positions else COLORS["storage"]
            x0, y0 = cell_top_left(row, col)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL, y0 + CELL), color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL, y0 + CELL), COLORS["grid"], 1)

    for area in layout.interaction_areas:
        for node_id in area.conflict_zone_nodes:
            draw_filled_node(frame, layout, node_id, COLORS["conflict"], "C")
        for node_id in area.waiting_points:
            draw_filled_node(frame, layout, node_id, COLORS["waiting"], "W")

    for station_name, node_id in layout.stations.items():
        color = COLORS["start"] if station_name.endswith("START") else COLORS["exit"]
        draw_station(frame, layout, node_id, color)

    return frame


def draw_filled_node(frame: np.ndarray, layout, node_id: str, color: tuple[int, int, int], label: str) -> None:
    row, col = layout.graph.nodes[node_id].position
    x0, y0 = cell_top_left(row, col)
    cv2.rectangle(frame, (x0 + 1, y0 + 1), (x0 + CELL - 1, y0 + CELL - 1), color, -1)
    cv2.putText(
        frame,
        label,
        (x0 + 7, y0 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        COLORS["text"],
        1,
        cv2.LINE_AA,
    )


def draw_station(frame: np.ndarray, layout, node_id: str, color: tuple[int, int, int]) -> None:
    row, col = layout.graph.nodes[node_id].position
    x0, y0 = cell_top_left(row, col)
    inset = 4
    cv2.rectangle(
        frame,
        (x0 + inset, y0 + inset),
        (x0 + CELL - inset, y0 + CELL - inset),
        color,
        2,
    )


def cell_top_left(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * CELL, MARGIN + row * CELL


if __name__ == "__main__":
    main()
