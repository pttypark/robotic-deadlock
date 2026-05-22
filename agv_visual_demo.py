"""Tkinter visual demo for the AGV interaction-area foundation layer."""

from __future__ import annotations

import argparse
import os
import sys
import tkinter as tk
from tkinter import ttk

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.agv_layout import build_four_way_intersection_layout
from rware.agv_movement import AGVMovementSimulator
from rware.agv_types import AGVAction


CELL = 62
MARGIN = 38
NODE_COLORS = {
    "normal": "#d6d8dc",
    "waiting": "#f4a742",
    "conflict": "#e04f5f",
    "goal": "#5abf69",
}
AGV_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#8c564b"}


class AGVVisualDemo:
    """Animate AGVs moving through the four-way interaction area."""

    def __init__(self, root: tk.Tk, delay_ms: int) -> None:
        """Initialize widgets and simulator state."""

        self.root = root
        self.delay_ms = delay_ms
        self.running = False
        self.layout = build_four_way_intersection_layout()
        self.sim = AGVMovementSimulator(self.layout)
        self.rows = max(node.position[0] for node in self.layout.graph.nodes.values()) + 1
        self.cols = max(node.position[1] for node in self.layout.graph.nodes.values()) + 1
        self._build_widgets()
        self._draw()

    def _build_widgets(self) -> None:
        self.root.title("AGV Interaction Area Demo")
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        width = self.cols * CELL + 2 * MARGIN
        height = self.rows * CELL + 2 * MARGIN
        self.canvas = tk.Canvas(
            outer,
            width=width,
            height=height,
            background="#f7f7f7",
            highlightthickness=1,
            highlightbackground="#b8bdc4",
        )
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="nsew")

        side = ttk.Frame(outer, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="nsew")
        controls = ttk.Frame(side)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=0, padx=2)
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(controls, text="Step", command=self.step_once).grid(row=0, column=2, padx=2)
        ttk.Button(controls, text="Reset", command=self.reset).grid(row=0, column=3, padx=2)

        self.summary = tk.Text(side, width=38, height=36, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def start(self) -> None:
        """Start automatic animation."""

        if not self.running:
            self.running = True
            self._schedule_next()

    def pause(self) -> None:
        """Pause automatic animation."""

        self.running = False

    def reset(self) -> None:
        """Reset the simulator and redraw."""

        self.running = False
        self.sim = AGVMovementSimulator(self.layout)
        self._draw()

    def step_once(self) -> None:
        """Advance one scripted step and redraw."""

        if self.sim.step_count > 0 and self.sim.step_count % 8 == 0:
            self.sim = AGVMovementSimulator(self.layout)
        self.sim.step(scripted_actions(self.sim.step_count))
        self._draw()

    def _schedule_next(self) -> None:
        if self.running:
            self.step_once()
            self.root.after(self.delay_ms, self._schedule_next)

    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_edges()
        self._draw_nodes()
        self._draw_agvs()
        self._draw_summary()

    def _draw_grid(self) -> None:
        for row in range(self.rows):
            for col in range(self.cols):
                x0, y0 = cell_top_left(row, col)
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + CELL,
                    y0 + CELL,
                    fill="#ffffff",
                    outline="#e1e4e8",
                )

    def _draw_edges(self) -> None:
        for edge in self.layout.graph.edges.values():
            start = self.layout.graph.nodes[edge.from_node].position
            target = self.layout.graph.nodes[edge.to_node].position
            x0, y0 = cell_center(*start)
            x1, y1 = cell_center(*target)
            self.canvas.create_line(
                x0,
                y0,
                x1,
                y1,
                arrow=tk.LAST,
                fill="#6b7280",
                width=3,
                arrowshape=(10, 12, 4),
            )

    def _draw_nodes(self) -> None:
        for node in self.layout.graph.nodes.values():
            x, y = cell_center(*node.position)
            radius = 16 if node.node_type == "conflict" else 13
            self.canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=NODE_COLORS.get(node.node_type, "#d6d8dc"),
                outline="#1f2937",
                width=2,
            )
            self.canvas.create_text(
                x,
                y + 25,
                text=short_label(node.node_id),
                fill="#111827",
                font=("Segoe UI", 8),
            )

    def _draw_agvs(self) -> None:
        by_node = {}
        for agv in self.sim.agvs:
            by_node.setdefault(agv.current_node, []).append(agv)

        for node_id, agvs in by_node.items():
            row, col = self.layout.graph.nodes[node_id].position
            base_x, base_y = cell_center(row, col)
            for agv, (dx, dy) in zip(agvs, spread_offsets(len(agvs))):
                x = base_x + dx
                y = base_y + dy
                color = AGV_COLORS.get(agv.agv_id, "#111827")
                self.canvas.create_oval(
                    x - 12,
                    y - 12,
                    x + 12,
                    y + 12,
                    fill=color,
                    outline="#ffffff",
                    width=2,
                )
                self.canvas.create_text(
                    x,
                    y,
                    text=str(agv.agv_id),
                    fill="#ffffff",
                    font=("Segoe UI", 9, "bold"),
                )
                hx, hy = heading_delta(agv.heading.name)
                self.canvas.create_line(
                    x,
                    y,
                    x + hx * 19,
                    y + hy * 19,
                    arrow=tk.LAST,
                    fill=color,
                    width=3,
                    arrowshape=(8, 10, 4),
                )

    def _draw_summary(self) -> None:
        snapshot = self.sim.debug_snapshot()
        lines = [
            f"Step: {self.sim.step_count}",
            f"Nodes: {snapshot['node_count']}",
            f"Edges: {snapshot['edge_count']}",
            f"Interaction Areas: {snapshot['interaction_area_count']}",
            "",
            "Legend",
            "orange: Waiting Point",
            "red: Conflict Point",
            "green: Goal",
            "",
            "AGV State",
        ]
        for agv in snapshot["agvs"]:
            lines.append(
                f"AGV {agv['agv_id']} node={agv['current_node']} "
                f"heading={agv['heading']} priority={agv['priority']}"
            )
            flags = []
            if agv["is_waiting_at_waiting_point"]:
                flags.append("WP")
            if agv["occupied_conflict_point"]:
                flags.append("CP")
            if flags:
                lines.append(f"  state={','.join(flags)}")

        lines.extend(["", "Collision Counters"])
        for key, value in snapshot["collision_counters"].items():
            lines.append(f"{key}: {value}")

        lines.extend(["", "Recent Events"])
        for event in snapshot["collision_events"][-8:]:
            lines.append(
                f"s{event['step']} {event['type']} robots={event['robots']} "
                f"node={event['node']} edge={event['edge']}"
            )

        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, "\n".join(lines))


def scripted_actions(step: int) -> dict[int, AGVAction]:
    """Return deterministic actions for the demo scenario."""

    phase = step % 8
    if phase in {0, 1, 2, 3}:
        return {agv_id: AGVAction.FORWARD for agv_id in (1, 2, 3, 4)}
    if phase == 4:
        return {agv_id: AGVAction.TURN_LEFT for agv_id in (1, 2, 3, 4)}
    return {agv_id: AGVAction.WAIT for agv_id in (1, 2, 3, 4)}


def cell_top_left(row: int, col: int) -> tuple[int, int]:
    """Return top-left pixel coordinates for one grid cell."""

    return MARGIN + col * CELL, MARGIN + row * CELL


def cell_center(row: int, col: int) -> tuple[int, int]:
    """Return center pixel coordinates for one grid cell."""

    x0, y0 = cell_top_left(row, col)
    return x0 + CELL // 2, y0 + CELL // 2


def short_label(node_id: str) -> str:
    """Return a compact label for one node id."""

    return (
        node_id.replace("CP_CENTER", "CP")
        .replace("_AP", " AP")
        .replace("_WP", " WP")
        .replace("_EXIT", " G")
    )


def heading_delta(heading_name: str) -> tuple[int, int]:
    """Return a drawing direction for one heading."""

    return {
        "NORTH": (0, -1),
        "EAST": (1, 0),
        "SOUTH": (0, 1),
        "WEST": (-1, 0),
    }[heading_name]


def spread_offsets(count: int) -> list[tuple[int, int]]:
    """Offset AGVs that occupy the same node so all remain visible."""

    if count == 1:
        return [(0, 0)]
    if count == 2:
        return [(-10, -10), (10, 10)]
    if count == 3:
        return [(-12, -12), (12, -12), (0, 12)]
    return [(-12, -12), (12, -12), (-12, 12), (12, 12)]


def run_headless(steps: int) -> None:
    """Run the same scripted demo without opening a GUI."""

    sim = AGVMovementSimulator(build_four_way_intersection_layout())
    for _ in range(steps):
        result = sim.step(scripted_actions(sim.step_count))
        print(f"step={result['step']} events={result['conflict_events']}")
    sim.print_summary()


def main() -> None:
    """Parse CLI arguments and launch the visual demo."""

    parser = argparse.ArgumentParser(description="Visualize the AGV intersection demo.")
    parser.add_argument("--delay-ms", type=int, default=850)
    parser.add_argument("--headless-steps", type=int, default=0)
    args = parser.parse_args()

    if args.headless_steps:
        run_headless(args.headless_steps)
        return

    root = tk.Tk()
    AGVVisualDemo(root, args.delay_ms)
    root.mainloop()


if __name__ == "__main__":
    main()
