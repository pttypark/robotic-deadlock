"""Tkinter visualization for the 12-AGV A* + FCFS cross experiment."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tkinter as tk
from tkinter import ttk

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment


CELL = 44
MARGIN = 26
COLORS = {
    "background": "#f7f7f7",
    "blocked": "#d8dde2",
    "road": "#fffdf8",
    "start": "#c8f2cf",
    "exit": "#ff2020",
    "conflict": "#f4ddcf",
    "attention": "#95d5ff",
    "waiting": "#ffd27a",
    "grid": "#222222",
}
ROBOT_COLORS = {
    "NORTH": "#1f77b4",
    "SOUTH": "#2ca02c",
    "WEST": "#9467bd",
    "EAST": "#ff7f0e",
}


class FCFSCrossVisualizer:
    """Visualize the FCFS shared-area admission baseline."""

    def __init__(self, root: tk.Tk, delay_ms: int = 600, random_seed: int = 7) -> None:
        """Initialize Tkinter widgets and experiment state."""

        self.root = root
        self.delay_ms = delay_ms
        self.random_seed = random_seed
        self.running = False
        self.experiment = FCFSCrossExperiment(robots_per_direction=3, random_seed=random_seed)
        self.rows, self.cols = self.experiment.layout.grid_size
        self._build_widgets()
        self._draw()

    def _build_widgets(self) -> None:
        self.root.title("A* + FCFS Cross Shared-Area Experiment")
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        width = self.cols * CELL + 2 * MARGIN
        height = self.rows * CELL + 2 * MARGIN
        self.canvas = tk.Canvas(
            outer,
            width=width,
            height=height,
            background=COLORS["background"],
            highlightthickness=1,
            highlightbackground="#999999",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")

        side = ttk.Frame(outer, padding=(12, 0, 0, 0))
        side.grid(row=0, column=1, sticky="nsew")
        controls = ttk.Frame(side)
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=0, padx=2)
        ttk.Button(controls, text="Pause", command=self.pause).grid(row=0, column=1, padx=2)
        ttk.Button(controls, text="Step", command=self.step_once).grid(row=0, column=2, padx=2)
        ttk.Button(controls, text="Reset", command=self.reset).grid(row=0, column=3, padx=2)

        self.summary = tk.Text(side, width=44, height=34, wrap=tk.WORD)
        self.summary.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def start(self) -> None:
        """Start automatic stepping."""

        if not self.running:
            self.running = True
            self._schedule_next()

    def pause(self) -> None:
        """Pause automatic stepping."""

        self.running = False

    def reset(self) -> None:
        """Reset the experiment."""

        self.running = False
        self.experiment = FCFSCrossExperiment(robots_per_direction=3, random_seed=self.random_seed)
        self._draw()

    def step_once(self) -> None:
        """Advance one simulation step."""

        if not self.experiment.is_done:
            self.experiment.step()
        self._draw()

    def _schedule_next(self) -> None:
        if not self.running:
            return
        self.step_once()
        if self.experiment.is_done:
            self.running = False
            return
        self.root.after(self.delay_ms, self._schedule_next)

    def _draw(self) -> None:
        self.canvas.delete("all")
        self._draw_cells()
        self._draw_pending_queues()
        self._draw_robots()
        self._draw_summary()

    def _draw_cells(self) -> None:
        graph = self.experiment.graph
        road_positions = {node.position for node in graph.nodes.values()}
        start_nodes = set(self.experiment.START_BY_DIRECTION.values())
        exit_nodes = set(self.experiment.EXIT_BY_DIRECTION.values())
        for row in range(self.rows):
            for col in range(self.cols):
                x0, y0 = cell_top_left(row, col)
                color = COLORS["blocked"]
                node_id = graph.position_to_node.get((row, col))
                if (row, col) in road_positions:
                    color = COLORS["road"]
                if node_id in start_nodes:
                    color = COLORS["start"]
                elif node_id in exit_nodes:
                    color = COLORS["exit"]
                elif node_id in self.experiment.area.conflict_zone_nodes:
                    color = COLORS["conflict"]
                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x0 + CELL,
                    y0 + CELL,
                    fill=color,
                    outline=COLORS["grid"],
                    width=1,
                )
                if node_id in start_nodes:
                    self._cell_label(row, col, "S", "#0b6b20")
                elif node_id in exit_nodes:
                    self._cell_label(row, col, "E", "#ffffff")
                elif node_id in self.experiment.area.waiting_points:
                    self._circle_label(row, col, "W", COLORS["waiting"])
                elif node_id in self.experiment.area.attention_points:
                    self._circle_label(row, col, "A", COLORS["attention"])
                elif node_id in self.experiment.area.conflict_zone_nodes:
                    self._cell_label(row, col, "C", "#6b2b1c")

    def _draw_pending_queues(self) -> None:
        # Pending AGVs are drawn outside each start cell to show 3 vehicles per direction.
        offsets = {
            "NORTH": (0, -1),
            "SOUTH": (0, 1),
            "WEST": (-1, 0),
            "EAST": (1, 0),
        }
        for direction, queue in self.experiment.pending_by_direction.items():
            start_node = self.experiment.START_BY_DIRECTION[direction]
            row, col = self.experiment.graph.nodes[start_node].position
            dx, dy = offsets[direction]
            for idx, robot in enumerate(queue):
                x, y = cell_center(row, col)
                x += dx * (22 + idx * 18)
                y += dy * (22 + idx * 18)
                self._draw_robot_at(x, y, robot.robot_id, direction)

    def _draw_robots(self) -> None:
        by_node = {}
        for robot in self.experiment.active.values():
            by_node.setdefault(robot.current_node, []).append(robot)
        for robot in self.experiment.completed:
            by_node.setdefault(robot.goal_node, []).append(robot)
        for node_id, robots in by_node.items():
            row, col = self.experiment.graph.nodes[node_id].position
            base_x, base_y = cell_center(row, col)
            for robot, (dx, dy) in zip(robots, spread_offsets(len(robots))):
                completed = robot.status == "completed"
                self._draw_robot_at(base_x + dx, base_y + dy, robot.robot_id, robot.direction, completed=completed)

    def _draw_robot_at(self, x: int, y: int, robot_id: str, direction: str, completed: bool = False) -> None:
        color = ROBOT_COLORS[direction]
        outline = "#111111" if completed else "#ffffff"
        width = 2 if completed else 2
        self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill=color, outline=outline, width=width)
        self.canvas.create_text(x, y, text=robot_id[-1], fill="#ffffff", font=("Segoe UI", 9, "bold"))

    def _draw_summary(self) -> None:
        metrics = self.experiment.metrics()
        lines = [
            "A* + FCFS Baseline",
            f"Step: {self.experiment.step_count}",
            f"Total AGVs: {metrics['robots']}",
            f"Random seed: {metrics['random_seed']}",
            f"Completed: {metrics['completed']}",
            f"TOTAL time: {metrics['total_time'] if self.experiment.is_done else '-'}",
            "",
            "Rule",
            "1 AGV only in shared conflict zone",
            "Decision at W point by FCFS",
            "",
            f"Shared owner: {self.experiment.shared_robot_id}",
            f"FCFS queue: {list(self.experiment.fcfs_queue)}",
            "",
            "Legend",
            "Gray: blocked / inaccessible",
            "White: AGV lane",
            "S: start",
            "E: exit",
            "A: attention point",
            "W: waiting / decision point",
            "C: conflict zone",
            "AGV at E: completed AGV",
            "",
            "Recent events",
        ]
        for event in self.experiment.event_log[-10:]:
            lines.append(f"s{event['step']} {event['event_type']} {event.get('robot_id', '')} {event.get('node', '')}")
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, "\n".join(lines))

    def _cell_label(self, row: int, col: int, text: str, color: str) -> None:
        x, y = cell_center(row, col)
        self.canvas.create_text(x, y, text=text, fill=color, font=("Segoe UI", 12, "bold"))

    def _circle_label(self, row: int, col: int, text: str, color: str) -> None:
        x, y = cell_center(row, col)
        self.canvas.create_oval(x - 12, y - 12, x + 12, y + 12, fill=color, outline="#ffffff", width=1)
        self.canvas.create_text(x, y, text=text, fill="#ffffff", font=("Segoe UI", 9, "bold"))


def cell_top_left(row: int, col: int) -> tuple[int, int]:
    """Return the top-left pixel for one cell."""

    return MARGIN + col * CELL, MARGIN + row * CELL


def cell_center(row: int, col: int) -> tuple[int, int]:
    """Return the center pixel for one cell."""

    x0, y0 = cell_top_left(row, col)
    return x0 + CELL // 2, y0 + CELL // 2


def spread_offsets(count: int) -> list[tuple[int, int]]:
    """Return small offsets for robots sharing one visual cell."""

    if count == 1:
        return [(0, 0)]
    if count == 2:
        return [(-10, -10), (10, 10)]
    if count == 3:
        return [(-12, -12), (12, -12), (0, 12)]
    if count == 4:
        return [(-12, -12), (12, -12), (-12, 12), (12, 12)]
    radius = 14
    return [
        (
            int(math.cos(2 * math.pi * idx / count) * radius),
            int(math.sin(2 * math.pi * idx / count) * radius),
        )
        for idx in range(count)
    ]


def main() -> None:
    """Parse arguments and start the visualizer."""

    parser = argparse.ArgumentParser(description="Visualize FCFS cross experiment.")
    parser.add_argument("--delay-ms", type=int, default=600)
    parser.add_argument("--random-seed", type=int, default=7)
    args = parser.parse_args()

    root = tk.Tk()
    FCFSCrossVisualizer(root, delay_ms=args.delay_ms, random_seed=args.random_seed)
    root.mainloop()


if __name__ == "__main__":
    main()
