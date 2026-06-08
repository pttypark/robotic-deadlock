from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from rware.agv_layouts import build_fcfs_double_cross_shared_area_layout  # noqa: E402
from rware.fcfs_cross_simulation import FCFSCrossExperiment  # noqa: E402
from scripts.run_total_time_bo_policy_comparison import _complete_heuristic_weights  # noqa: E402


DEFAULT_SELECTION_FILE = (
    PROJECT_DIR
    / "final_output"
    / "layout_policy_bo_advantage_analysis"
    / "selected_bo_advantage_cases.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "final_output"
    / "layout_policy_bo_advantage_analysis"
    / "selected_3d_videos"
)
DEFAULT_DESKTOP_DIR = Path.home() / "OneDrive" / "바탕 화면" / "selected_bo_advantage_3d_videos"
SINGLE_RESULT_DIR = PROJECT_DIR / "final_output" / "test_100_by_agv_fixed_weights"
DOUBLE_RESULT_DIR = PROJECT_DIR / "final_output" / "double_cross_test_100_by_agv_fixed_weights"

POLICIES = ("fcfs", "fixed_heuristic", "bo_heuristic", "pso_heuristic")
POLICY_LABELS = {
    "fcfs": "FCFS",
    "fixed_heuristic": "Basic Heuristic",
    "bo_heuristic": "BO Heuristic",
    "pso_heuristic": "PSO Heuristic",
}
POLICY_WEIGHT_KEYS = {
    "fixed_heuristic": "basic_heuristic",
    "bo_heuristic": "bo_heuristic",
    "pso_heuristic": "pso_heuristic",
}
BASIC_WEIGHTS = {
    "waiting": 5.985670,
    "maneuver": -3.799914,
    "exit_competition": -0.699648,
    "path_conflict": -0.544682,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}
BO_WEIGHTS = {
    "waiting": 5.660327,
    "maneuver": -4.0,
    "exit_competition": -4.0,
    "path_conflict": -0.656567,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}
PSO_WEIGHTS = {
    "waiting": 8.339856,
    "maneuver": -3.688935,
    "exit_competition": -2.609410,
    "path_conflict": -3.0,
    "approach_queue": 0.0,
    "remaining_path": 0.0,
    "same_direction_backlog": 0.0,
}
WEIGHTS_BY_POLICY = {
    "fixed_heuristic": _complete_heuristic_weights(BASIC_WEIGHTS),
    "bo_heuristic": _complete_heuristic_weights(BO_WEIGHTS),
    "pso_heuristic": _complete_heuristic_weights(PSO_WEIGHTS),
}
COLORS = {
    "background": (230, 234, 234),
    "floor_dark": (197, 203, 205),
    "floor_line": (177, 185, 188),
    "road": (235, 238, 236),
    "road_edge": (139, 149, 153),
    "lane_mark": (255, 255, 247),
    "waiting": (66, 156, 196),
    "conflict": (62, 78, 124),
    "start": (103, 176, 128),
    "exit": (221, 124, 104),
    "text": (30, 34, 40),
    "muted": (85, 93, 101),
    "panel": (24, 30, 40),
    "panel_light": (246, 248, 247),
    "shadow": (35, 39, 45),
    "metal": (58, 65, 72),
    "tire": (22, 25, 28),
    "headlight": (246, 249, 204),
    "progress": (86, 181, 130),
    "queue": (255, 190, 72),
    "shared": (76, 194, 222),
    "bo": (58, 154, 255),
}
ROBOT_COLORS = {
    "NORTH": (42, 114, 184),
    "L_NORTH": (42, 114, 184),
    "R_NORTH": (42, 114, 184),
    "SOUTH": (66, 142, 74),
    "L_SOUTH": (66, 142, 74),
    "R_SOUTH": (66, 142, 74),
    "WEST": (172, 80, 112),
    "EAST": (196, 118, 48),
}
POLICY_ACCENTS = {
    "fcfs": (118, 130, 145),
    "fixed_heuristic": (77, 169, 108),
    "bo_heuristic": (58, 154, 255),
    "pso_heuristic": (221, 95, 91),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record 3D videos for BO-advantage selected single/double-cross scenarios."
    )
    parser.add_argument("--selection-file", type=Path, default=DEFAULT_SELECTION_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-to-desktop", action="store_true")
    parser.add_argument("--desktop-dir", type=Path, default=DEFAULT_DESKTOP_DIR)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--frames-per-step", type=int, default=3)
    parser.add_argument("--width", type=int, default=1720)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = _read_csv(args.selection_file)
    video_rows = []

    for selected in selected_rows:
        layout_key = selected["layout_key"]
        scenario = _load_selected_scenario(layout_key, int(selected["run_index"]))
        for policy in POLICIES:
            path, metrics = _record_policy_video(
                layout_key=layout_key,
                policy=policy,
                scenario=scenario,
                selected=selected,
                args=args,
            )
            video_rows.append(
                {
                    "layout_key": layout_key,
                    "run_index": scenario["run_index"],
                    "scenario_case_id": scenario["scenario_case_id"],
                    "scenario_case_name": scenario["scenario_case_name"],
                    "total_agvs": scenario["total_agvs"],
                    "policy": policy,
                    "total_time": metrics["total_time"],
                    "completed": metrics["completed"],
                    "video_path": str(path),
                }
            )
            print(f"{layout_key} {policy}: {path} total_time={metrics['total_time']}")

    _write_csv(video_rows, args.output_dir / "selected_3d_video_outputs.csv")
    _write_readme(video_rows, selected_rows, args.output_dir)

    if args.copy_to_desktop:
        args.desktop_dir.mkdir(parents=True, exist_ok=True)
        for path in args.output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, args.desktop_dir / path.name)
        print(f"copied videos to desktop: {args.desktop_dir}")

    print(f"wrote videos: {args.output_dir.resolve()}")


def _load_selected_scenario(layout_key: str, run_index: int) -> dict:
    if layout_key == "single_cross":
        scenario_file = SINGLE_RESULT_DIR / "test_scenarios_300_all_agv_counts.csv"
        row = next(row for row in _read_csv(scenario_file) if int(row["run_index"]) == run_index)
        allocation = {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
    elif layout_key == "double_cross":
        scenario_file = DOUBLE_RESULT_DIR / "double_cross_test_scenarios_200_all_agv_counts.csv"
        row = next(row for row in _read_csv(scenario_file) if int(row["run_index"]) == run_index)
        allocation = {
            "L_NORTH": int(row["l_north_agvs"]),
            "L_SOUTH": int(row["l_south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "R_NORTH": int(row["r_north_agvs"]),
            "R_SOUTH": int(row["r_south_agvs"]),
            "EAST": int(row["east_agvs"]),
        }
    else:
        raise ValueError(f"Unknown layout_key: {layout_key}")
    return {
        "layout_key": layout_key,
        "run_index": int(row["run_index"]),
        "seed": int(row["scenario_seed"]),
        "scenario_case_id": int(row["scenario_case_id"]),
        "scenario_case_name": row["scenario_case_name"],
        "scenario_case_label": row.get("scenario_case_label", ""),
        "total_agvs": int(row["total_agvs"]),
        "allocation": allocation,
        "spawn_plan": json.loads(row["spawn_plan_json"]),
        "goal_plan": json.loads(row["goal_plan_json"]),
    }


def _record_policy_video(
    layout_key: str,
    policy: str,
    scenario: dict,
    selected: dict,
    args: argparse.Namespace,
) -> tuple[Path, dict]:
    experiment = _build_experiment(layout_key, policy, scenario, args)
    snapshots = [_snapshot(experiment)]
    while not experiment.is_done and experiment.step_count < args.max_steps:
        experiment.step()
        snapshots.append(_snapshot(experiment))
    metrics = experiment.metrics()
    path = (
        args.output_dir
        / f"{layout_key}_run{scenario['run_index']:03d}_{policy}_3d_total{metrics['total_time']}.mp4"
    )
    _write_video(
        path=path,
        snapshots=snapshots,
        experiment=experiment,
        policy=policy,
        metrics=metrics,
        selected=selected,
        fps=args.fps,
        frames_per_step=args.frames_per_step,
        width=args.width,
        height=args.height,
    )
    return path, metrics


def _build_experiment(
    layout_key: str,
    policy: str,
    scenario: dict,
    args: argparse.Namespace,
) -> FCFSCrossExperiment:
    layout = None
    if layout_key == "double_cross":
        layout = build_fcfs_double_cross_shared_area_layout(corridor_length=args.corridor_length)
    policy_type = "fcfs" if policy == "fcfs" else "heuristic"
    return FCFSCrossExperiment(
        layout=layout,
        robots_by_direction=scenario["allocation"],
        random_seed=scenario["seed"],
        corridor_length=args.corridor_length,
        west_exit_extension=0,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario["spawn_plan"],
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=args.shared_area_capacity,
        normalize_heuristic_features=True,
        goal_plan_by_direction=scenario["goal_plan"],
        policy_type=policy_type,
        heuristic_weights=WEIGHTS_BY_POLICY.get(policy) if policy_type == "heuristic" else None,
    )


def _snapshot(experiment: FCFSCrossExperiment) -> dict:
    shared_robot_ids_by_area = {
        area_id: sorted(robot_ids)
        for area_id, robot_ids in experiment.shared_robot_ids_by_area.items()
    }
    queued_robot_ids = sorted(
        {
            robot_id
            for queue in experiment.fcfs_queues.values()
            for robot_id in queue
        }
    )
    shared_robot_ids = sorted(
        {
            robot_id
            for robot_ids in shared_robot_ids_by_area.values()
            for robot_id in robot_ids
        }
    )
    robots = {}
    for robot_id, robot in experiment.active.items():
        robots[robot_id] = {
            "node": robot.current_node,
            "direction": robot.direction,
            "status": robot.status,
            "in_queue": robot_id in queued_robot_ids,
            "in_shared": robot_id in shared_robot_ids,
        }
    for robot in experiment.completed:
        robots[robot.robot_id] = {
            "node": robot.goal_node,
            "direction": robot.direction,
            "status": "completed",
            "in_queue": False,
            "in_shared": False,
        }
    queues = {
        area_id: list(queue)
        for area_id, queue in experiment.fcfs_queues.items()
    }
    return {
        "step": experiment.step_count,
        "robots": robots,
        "shared_robot_ids": shared_robot_ids,
        "shared_robot_ids_by_area": shared_robot_ids_by_area,
        "queued_robot_ids": queued_robot_ids,
        "queues": queues,
    }


def _write_video(
    path: Path,
    snapshots: list[dict],
    experiment: FCFSCrossExperiment,
    policy: str,
    metrics: dict,
    selected: dict,
    fps: int,
    frames_per_step: int,
    width: int,
    height: int,
) -> None:
    width += width % 2
    height += height % 2
    renderer = IsoRenderer(experiment, width=width, height=height)
    base = renderer.base_frame()
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    subframes = max(1, frames_per_step)
    for index in range(len(snapshots) - 1):
        left = snapshots[index]
        right = snapshots[index + 1]
        for subframe in range(subframes):
            alpha = _smoothstep(subframe / subframes)
            frame = base.copy()
            renderer.draw_recent_trails(frame, snapshots, index, alpha)
            renderer.draw_zone_status(frame, left)
            renderer.draw_robots(frame, left, right, alpha)
            renderer.draw_overlay(frame, policy, metrics, selected, left, alpha)
            writer.write(frame)
    frame = base.copy()
    renderer.draw_recent_trails(frame, snapshots, len(snapshots) - 1, 1.0)
    renderer.draw_zone_status(frame, snapshots[-1])
    renderer.draw_robots(frame, snapshots[-1], snapshots[-1], 1.0)
    renderer.draw_overlay(frame, policy, metrics, selected, snapshots[-1], 1.0)
    writer.write(frame)
    writer.release()


class IsoRenderer:
    def __init__(self, experiment: FCFSCrossExperiment, width: int, height: int) -> None:
        self.experiment = experiment
        self.width = width
        self.height = height
        rows, cols = experiment.layout.grid_size
        self.scale_x = min(42.0, width * 0.92 / max(1, rows + cols))
        self.scale_y = min(23.0, height * 0.84 / max(1, rows + cols))
        self.scale_z = self.scale_y * 3.2
        self.origin_x = width / 2
        self.origin_y = 70.0
        self.road_positions = {node.position for node in experiment.graph.nodes.values()}
        self.start_nodes = set(experiment.START_BY_DIRECTION.values())
        self.exit_nodes = set(experiment.EXIT_DIRECTION_BY_NODE)
        self.conflict_nodes = set().union(*(area.conflict_zone_nodes for area in experiment.areas))
        self.waiting_nodes = set().union(*(area.waiting_points for area in experiment.areas))

    def base_frame(self) -> np.ndarray:
        frame = self.floor_frame()
        self.draw_floor_grid(frame)
        for row, col in sorted(self.road_positions, key=lambda pos: (pos[0] + pos[1], pos[0])):
            node_id = self.experiment.graph.position_to_node.get((row, col))
            color = COLORS["road"]
            label = ""
            if node_id in self.start_nodes:
                color = COLORS["start"]
                label = "S"
            elif node_id in self.exit_nodes:
                color = COLORS["exit"]
                label = "E"
            elif node_id in self.conflict_nodes:
                color = COLORS["conflict"]
                label = "C"
            elif node_id in self.waiting_nodes:
                color = COLORS["waiting"]
                label = "W"
            self.draw_tile(frame, row, col, color, label)
        for area in self.experiment.areas:
            self.draw_shared_zone_outline(frame, area)
        self.draw_port_labels(frame)
        return frame

    def floor_frame(self) -> np.ndarray:
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for y in range(self.height):
            ratio = y / max(1, self.height - 1)
            color = _blend(COLORS["background"], COLORS["floor_dark"], ratio * 0.45)
            frame[y, :, :] = color
        return frame

    def draw_floor_grid(self, frame: np.ndarray) -> None:
        for offset in range(-self.height, self.width + self.height, 92):
            cv2.line(frame, (offset, 0), (offset + self.height, self.height), COLORS["floor_line"], 1, cv2.LINE_AA)
            cv2.line(frame, (offset, self.height), (offset + self.height, 0), COLORS["floor_line"], 1, cv2.LINE_AA)

    def draw_tile(self, frame: np.ndarray, row: float, col: float, color: tuple[int, int, int], label: str) -> None:
        points = np.array(
            [
                self.project(col - 0.5, row - 0.5, 0.0),
                self.project(col + 0.5, row - 0.5, 0.0),
                self.project(col + 0.5, row + 0.5, 0.0),
                self.project(col - 0.5, row + 0.5, 0.0),
            ],
            dtype=np.int32,
        )
        lower = np.array(
            [
                self.project(col - 0.5, row - 0.5, -0.10),
                self.project(col + 0.5, row - 0.5, -0.10),
                self.project(col + 0.5, row + 0.5, -0.10),
                self.project(col - 0.5, row + 0.5, -0.10),
            ],
            dtype=np.int32,
        )
        right_side = np.array([points[1], points[2], lower[2], lower[1]], dtype=np.int32)
        front_side = np.array([points[2], points[3], lower[3], lower[2]], dtype=np.int32)
        cv2.fillConvexPoly(frame, front_side, _blend(color, (0, 0, 0), 0.16), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, right_side, _blend(color, (0, 0, 0), 0.24), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, points, color, cv2.LINE_AA)
        cv2.polylines(frame, [points], True, COLORS["road_edge"], 1, cv2.LINE_AA)
        self.draw_lane_mark(frame, row, col)
        if label:
            x, y = self.project(col, row, 0.03)
            text_color = (255, 255, 255) if label in {"C", "E"} else COLORS["text"]
            cv2.putText(frame, label, (int(x) - 7, int(y) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    def draw_lane_mark(self, frame: np.ndarray, row: float, col: float) -> None:
        node_id = self.experiment.graph.position_to_node.get((int(row), int(col)))
        if node_id in self.conflict_nodes:
            return
        center = self.project(col, row, 0.04)
        vertical_neighbors = {
            self.experiment.graph.position_to_node.get((int(row) - 1, int(col))),
            self.experiment.graph.position_to_node.get((int(row) + 1, int(col))),
        }
        horizontal_neighbors = {
            self.experiment.graph.position_to_node.get((int(row), int(col) - 1)),
            self.experiment.graph.position_to_node.get((int(row), int(col) + 1)),
        }
        if any(neighbor in self.experiment.graph.nodes for neighbor in vertical_neighbors):
            p0 = self.project(col - 0.18, row, 0.045)
            p1 = self.project(col + 0.18, row, 0.045)
        elif any(neighbor in self.experiment.graph.nodes for neighbor in horizontal_neighbors):
            p0 = self.project(col, row - 0.18, 0.045)
            p1 = self.project(col, row + 0.18, 0.045)
        else:
            return
        cv2.line(frame, p0, p1, COLORS["lane_mark"], 1, cv2.LINE_AA)
        cv2.circle(frame, center, 1, _blend(COLORS["lane_mark"], COLORS["road"], 0.35), -1, cv2.LINE_AA)

    def draw_shared_zone_outline(self, frame: np.ndarray, area) -> None:
        conflict_positions = [self.experiment.graph.nodes[node_id].position for node_id in area.conflict_zone_nodes]
        min_row = min(row for row, _ in conflict_positions) - 0.55
        max_row = max(row for row, _ in conflict_positions) + 0.55
        min_col = min(col for _, col in conflict_positions) - 0.55
        max_col = max(col for _, col in conflict_positions) + 0.55
        points = np.array(
            [
                self.project(min_col, min_row, 0.07),
                self.project(max_col, min_row, 0.07),
                self.project(max_col, max_row, 0.07),
                self.project(min_col, max_row, 0.07),
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [points], True, (42, 55, 76), 3, cv2.LINE_AA)

    def draw_port_labels(self, frame: np.ndarray) -> None:
        for direction, node_id in self.experiment.START_BY_DIRECTION.items():
            if node_id not in self.experiment.graph.nodes:
                continue
            row, col = self.experiment.graph.nodes[node_id].position
            x, y = self.project(col, row, 0.42)
            self.draw_tag(frame, x, y - 24, f"IN {self.direction_short(direction)}", COLORS["start"])
        for node_id, direction in self.experiment.EXIT_DIRECTION_BY_NODE.items():
            if node_id not in self.experiment.graph.nodes:
                continue
            row, col = self.experiment.graph.nodes[node_id].position
            x, y = self.project(col, row, 0.42)
            self.draw_tag(frame, x, y - 24, f"OUT {self.direction_short(direction)}", COLORS["exit"])

    def draw_tag(self, frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.38
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        pad_x, pad_y = 8, 5
        x0 = int(x - text_width / 2 - pad_x)
        y0 = int(y - text_height - pad_y)
        x1 = int(x + text_width / 2 + pad_x)
        y1 = int(y + pad_y)
        self.alpha_rect(frame, x0, y0, x1, y1, _blend(color, (255, 255, 255), 0.20), 0.90, border=color)
        cv2.putText(frame, text, (x0 + pad_x, y1 - pad_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def direction_short(self, direction: str) -> str:
        return {
            "NORTH": "N",
            "SOUTH": "S",
            "WEST": "W",
            "EAST": "E",
            "L_NORTH": "L-N",
            "L_SOUTH": "L-S",
            "R_NORTH": "R-N",
            "R_SOUTH": "R-S",
        }.get(direction, direction[:3])

    def draw_recent_trails(self, frame: np.ndarray, snapshots: list[dict], index: int, alpha: float, max_history: int = 9) -> None:
        start = max(0, index - max_history)
        active = snapshots[index]["robots"]
        robot_ids = sorted(robot_id for robot_id, state in active.items() if state.get("status") != "completed")
        for robot_id in robot_ids:
            points: list[tuple[int, int]] = []
            for snap in snapshots[start:index + 1]:
                state = snap["robots"].get(robot_id)
                point = self.robot_screen_point(state)
                if point is not None:
                    points.append(point)
            if index + 1 < len(snapshots):
                left_state = snapshots[index]["robots"].get(robot_id)
                right_state = snapshots[index + 1]["robots"].get(robot_id)
                interp = self.interpolated_robot_point(left_state, right_state, alpha)
                if interp is not None:
                    points.append(interp)
            if len(points) < 2:
                continue
            color = ROBOT_COLORS.get(active[robot_id]["direction"], (70, 120, 170))
            for segment_index in range(1, len(points)):
                fade = segment_index / max(1, len(points) - 1)
                segment_color = _blend(COLORS["background"], color, 0.25 + 0.55 * fade)
                thickness = 1 + int(3 * fade)
                cv2.line(frame, points[segment_index - 1], points[segment_index], segment_color, thickness, cv2.LINE_AA)

    def robot_screen_point(self, state: dict | None) -> tuple[int, int] | None:
        if not state or state["node"] not in self.experiment.graph.nodes:
            return None
        row, col = self.experiment.graph.nodes[state["node"]].position
        return self.project(col, row, 0.42)

    def interpolated_robot_point(self, left_state: dict | None, right_state: dict | None, alpha: float) -> tuple[int, int] | None:
        if not left_state and not right_state:
            return None
        left_node = (left_state or right_state)["node"]
        right_node = (right_state or left_state)["node"]
        if left_node not in self.experiment.graph.nodes or right_node not in self.experiment.graph.nodes:
            return None
        left_row, left_col = self.experiment.graph.nodes[left_node].position
        right_row, right_col = self.experiment.graph.nodes[right_node].position
        row = left_row + (right_row - left_row) * alpha
        col = left_col + (right_col - left_col) * alpha
        return self.project(col, row, 0.42)

    def draw_zone_status(self, frame: np.ndarray, snapshot: dict) -> None:
        for index, area in enumerate(self.experiment.areas, start=1):
            conflict_positions = [self.experiment.graph.nodes[node_id].position for node_id in area.conflict_zone_nodes]
            center_row = sum(row for row, _ in conflict_positions) / len(conflict_positions)
            center_col = sum(col for _, col in conflict_positions) / len(conflict_positions)
            x, y = self.project(center_col, center_row, 0.88)
            shared_count = len(snapshot.get("shared_robot_ids_by_area", {}).get(area.area_id, []))
            queue_count = len(snapshot.get("queues", {}).get(area.area_id, []))
            text = f"Zone {index}: {shared_count}/{self.experiment.shared_area_capacity}  Q:{queue_count}"
            color = COLORS["shared"] if shared_count else COLORS["queue"] if queue_count else COLORS["muted"]
            self.draw_status_badge(frame, x, y - 44, text, color)

    def draw_status_badge(self, frame: np.ndarray, x: int, y: int, text: str, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.44
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, thickness)
        x0, y0 = x - text_width // 2 - 12, y - text_height - 8
        x1, y1 = x + text_width // 2 + 12, y + 8
        self.alpha_rect(frame, x0, y0, x1, y1, COLORS["panel"], 0.72, border=color)
        cv2.circle(frame, (x0 + 11, y0 + (y1 - y0) // 2), 4, color, -1, cv2.LINE_AA)
        cv2.putText(frame, text, (x0 + 22, y1 - 9), font, scale, (246, 248, 250), thickness, cv2.LINE_AA)

    def draw_robots(self, frame: np.ndarray, left: dict, right: dict, alpha: float) -> None:
        robot_ids = sorted(set(left["robots"]) | set(right["robots"]))
        placements = []
        for robot_id in robot_ids:
            left_state = left["robots"].get(robot_id)
            right_state = right["robots"].get(robot_id)
            state = right_state or left_state
            if not state:
                continue
            left_node = (left_state or right_state)["node"]
            right_node = (right_state or left_state)["node"]
            if left_node not in self.experiment.graph.nodes or right_node not in self.experiment.graph.nodes:
                continue
            left_row, left_col = self.experiment.graph.nodes[left_node].position
            right_row, right_col = self.experiment.graph.nodes[right_node].position
            row = left_row + (right_row - left_row) * alpha
            col = left_col + (right_col - left_col) * alpha
            placements.append((row + col, robot_id, row, col, right_row - left_row, right_col - left_col, state))
        for _, robot_id, row, col, drow, dcol, state in sorted(placements):
            self.draw_agv(frame, row=row, col=col, drow=drow, dcol=dcol, robot_id=robot_id, state=state)

    def draw_agv(self, frame: np.ndarray, row: float, col: float, drow: float, dcol: float, robot_id: str, state: dict) -> None:
        color = ROBOT_COLORS.get(state["direction"], (70, 120, 170))
        if state["status"] == "completed":
            color = _blend(color, (210, 210, 210), 0.45)
        length = 0.74
        width = 0.52
        height = 0.28
        if abs(dcol) >= abs(drow):
            x0, x1 = col - length / 2, col + length / 2
            y0, y1 = row - width / 2, row + width / 2
        else:
            x0, x1 = col - width / 2, col + width / 2
            y0, y1 = row - length / 2, row + length / 2
        z0, z1 = 0.08, 0.08 + height
        shadow = np.array(
            [
                self.project(x0 - 0.04, y0 + 0.08, 0.01),
                self.project(x1 - 0.04, y0 + 0.08, 0.01),
                self.project(x1 + 0.06, y1 + 0.10, 0.01),
                self.project(x0 + 0.06, y1 + 0.10, 0.01),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, shadow, _blend(COLORS["shadow"], COLORS["background"], 0.42), cv2.LINE_AA)
        top = np.array(
            [
                self.project(x0, y0, z1),
                self.project(x1, y0, z1),
                self.project(x1, y1, z1),
                self.project(x0, y1, z1),
            ],
            dtype=np.int32,
        )
        right_face = np.array([top[1], top[2], self.project(x1, y1, z0), self.project(x1, y0, z0)], dtype=np.int32)
        front_face = np.array([top[3], top[2], self.project(x1, y1, z0), self.project(x0, y1, z0)], dtype=np.int32)
        cv2.fillConvexPoly(frame, front_face, _blend(color, (0, 0, 0), 0.28), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, right_face, _blend(color, (0, 0, 0), 0.42), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, top, color, cv2.LINE_AA)
        cv2.polylines(frame, [top, right_face, front_face], True, (230, 235, 238), 1, cv2.LINE_AA)
        self.draw_agv_details(frame, row, col, x0, x1, y0, y1, z1, drow, dcol, color)
        if state.get("in_shared"):
            self.draw_status_ring(frame, row, col, z1, COLORS["shared"])
        elif state.get("in_queue"):
            self.draw_status_ring(frame, row, col, z1, COLORS["queue"])
        cx, cy = self.project(col, row, z1 + 0.02)
        label = _robot_number(robot_id)
        cv2.putText(frame, label, (int(cx) - 6 * len(label), int(cy) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_agv_details(self, frame: np.ndarray, row: float, col: float, x0: float, x1: float, y0: float, y1: float, z1: float, drow: float, dcol: float, color: tuple[int, int, int]) -> None:
        inset_x = (x1 - x0) * 0.22
        inset_y = (y1 - y0) * 0.22
        panel = np.array(
            [
                self.project(x0 + inset_x, y0 + inset_y, z1 + 0.018),
                self.project(x1 - inset_x, y0 + inset_y, z1 + 0.018),
                self.project(x1 - inset_x, y1 - inset_y, z1 + 0.018),
                self.project(x0 + inset_x, y1 - inset_y, z1 + 0.018),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, panel, _blend(color, (255, 255, 255), 0.18), cv2.LINE_AA)
        cv2.polylines(frame, [panel], True, _blend(color, (0, 0, 0), 0.40), 1, cv2.LINE_AA)
        if drow == 0 and dcol == 0:
            drow, dcol = 0.0, 1.0
        norm = math.hypot(drow, dcol) or 1.0
        hx, hy = self.project(col + 0.31 * dcol / norm, row + 0.31 * drow / norm, z1 + 0.035)
        cv2.circle(frame, (hx, hy), 3, COLORS["headlight"], -1, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), 5, _blend(COLORS["headlight"], color, 0.62), 1, cv2.LINE_AA)

    def draw_status_ring(self, frame: np.ndarray, row: float, col: float, z: float, color: tuple[int, int, int]) -> None:
        cx, cy = self.project(col, row, z + 0.12)
        cv2.circle(frame, (cx, cy), 19, _blend(color, (255, 255, 255), 0.08), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 24, _blend(color, COLORS["background"], 0.30), 1, cv2.LINE_AA)

    def draw_overlay(self, frame: np.ndarray, policy: str, metrics: dict, selected: dict, snapshot: dict, alpha: float) -> None:
        step = snapshot["step"] + alpha
        total_time = max(1.0, float(metrics["total_time"]))
        progress = min(1.0, step / total_time)
        completed_now = sum(1 for state in snapshot["robots"].values() if state.get("status") == "completed")
        robots_total = int(metrics["robots"])
        queued_count = len(snapshot.get("queued_robot_ids", []))
        shared_count = len(snapshot.get("shared_robot_ids", []))
        accent = POLICY_ACCENTS.get(policy, COLORS["muted"])

        self.alpha_rect(frame, 22, 20, 690, 166, COLORS["panel"], 0.78, border=accent)
        cv2.rectangle(frame, (22, 20), (31, 166), accent, -1)
        title = f"{POLICY_LABELS[policy]}  |  {selected['layout_key'].replace('_', ' ').title()}"
        cv2.putText(frame, title, (46, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (248, 250, 252), 2, cv2.LINE_AA)
        case_line = (
            f"run {selected['run_index']}  scenario {selected['scenario_case_id']} "
            f"({selected['scenario_case_name']})  AGV {robots_total}"
        )
        cv2.putText(frame, case_line, (46, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 222, 226), 1, cv2.LINE_AA)
        status_line = f"step {step:.1f}/{metrics['total_time']}   completed {completed_now}/{robots_total}   in-zone {shared_count}   queue {queued_count}"
        cv2.putText(frame, status_line, (46, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (215, 222, 226), 1, cv2.LINE_AA)
        self.draw_progress_bar(frame, 46, 126, 590, 15, progress, accent, label="time progress")
        completion = completed_now / max(1, robots_total)
        self.draw_progress_bar(frame, 46, 148, 590, 10, completion, COLORS["progress"], label="completion")

        margin = _safe_float(selected.get("bo_margin_vs_best_other"))
        if policy == "bo_heuristic" and margin is not None:
            self.draw_status_badge(frame, 540, 196, f"Selected BO case: -{margin:g} steps vs best other", COLORS["bo"])

        self.draw_policy_scoreboard(frame, policy, selected)
        self.draw_legend(frame)

    def draw_policy_scoreboard(self, frame: np.ndarray, current_policy: str, selected: dict) -> None:
        x0 = self.width - 330
        y0 = 24
        width = 304
        row_height = 34
        totals = {policy: _selected_policy_total(selected, policy) for policy in POLICIES}
        valid_totals = [value for value in totals.values() if value is not None]
        best_total = min(valid_totals) if valid_totals else None
        self.alpha_rect(frame, x0, y0, x0 + width, y0 + 184, COLORS["panel"], 0.74, border=(88, 96, 108))
        cv2.putText(frame, "Policy Total_Time", (x0 + 18, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (248, 250, 252), 2, cv2.LINE_AA)
        for row_index, policy in enumerate(POLICIES):
            y = y0 + 52 + row_index * row_height
            accent = POLICY_ACCENTS.get(policy, COLORS["muted"])
            if policy == current_policy:
                self.alpha_rect(frame, x0 + 12, y - 20, x0 + width - 12, y + 10, _blend(accent, COLORS["panel"], 0.40), 0.78)
            cv2.circle(frame, (x0 + 28, y - 5), 6, accent, -1, cv2.LINE_AA)
            label = POLICY_LABELS[policy].replace(" Heuristic", "")
            total = totals[policy]
            total_text = "-" if total is None else f"{total:g}"
            marker = " BEST" if best_total is not None and total == best_total else ""
            cv2.putText(frame, label, (x0 + 44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (232, 236, 239), 1, cv2.LINE_AA)
            cv2.putText(frame, total_text + marker, (x0 + 202, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (232, 236, 239), 1, cv2.LINE_AA)

    def draw_legend(self, frame: np.ndarray) -> None:
        x0 = 24
        y0 = self.height - 132
        self.alpha_rect(frame, x0, y0, x0 + 482, y0 + 104, COLORS["panel"], 0.62, border=(92, 101, 110))
        cv2.putText(frame, "Visual guide", (x0 + 16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)
        items = [
            ("IN", COLORS["start"]),
            ("OUT", COLORS["exit"]),
            ("Waiting", COLORS["waiting"]),
            ("Conflict", COLORS["conflict"]),
            ("Queue ring", COLORS["queue"]),
            ("In-zone ring", COLORS["shared"]),
        ]
        for index, (label, color) in enumerate(items):
            col = index % 3
            row = index // 3
            x = x0 + 18 + col * 150
            y = y0 + 52 + row * 28
            cv2.rectangle(frame, (x, y - 11), (x + 18, y + 7), color, -1)
            cv2.rectangle(frame, (x, y - 11), (x + 18, y + 7), (235, 238, 240), 1)
            cv2.putText(frame, label, (x + 26, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (224, 230, 234), 1, cv2.LINE_AA)

    def draw_progress_bar(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        ratio: float,
        color: tuple[int, int, int],
        label: str = "",
    ) -> None:
        ratio = max(0.0, min(1.0, ratio))
        cv2.rectangle(frame, (x, y), (x + width, y + height), (72, 80, 90), -1)
        cv2.rectangle(frame, (x, y), (x + int(width * ratio), y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (180, 188, 196), 1)
        if label:
            cv2.putText(frame, label, (x + width + 10, y + height), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (202, 209, 214), 1, cv2.LINE_AA)

    def alpha_rect(
        self,
        frame: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        alpha: float,
        border: tuple[int, int, int] | None = None,
    ) -> None:
        x0 = max(0, min(self.width - 1, int(x0)))
        x1 = max(0, min(self.width - 1, int(x1)))
        y0 = max(0, min(self.height - 1, int(y0)))
        y1 = max(0, min(self.height - 1, int(y1)))
        if x1 <= x0 or y1 <= y0:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
        if border is not None:
            cv2.rectangle(frame, (x0, y0), (x1, y1), border, 1)

    def project(self, x: float, y: float, z: float) -> tuple[int, int]:
        return (
            int(round(self.origin_x + (x - y) * self.scale_x)),
            int(round(self.origin_y + (x + y) * self.scale_y - z * self.scale_z)),
        )


def _robot_number(robot_id: str) -> str:
    digits = "".join(character for character in robot_id if character.isdigit())
    return digits[-2:] if len(digits) > 1 else digits or robot_id[-1]


def _safe_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selected_policy_total(selected: dict, policy: str) -> float | None:
    keys = {
        "fcfs": "fcfs_total_time",
        "fixed_heuristic": "fixed_heuristic_total_time",
        "bo_heuristic": "bo_heuristic_total_time",
        "pso_heuristic": "pso_heuristic_total_time",
    }
    return _safe_float(selected.get(keys[policy]))


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _blend(color: tuple[int, int, int], other: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(int(round(color[index] * (1.0 - ratio) + other[index] * ratio)) for index in range(3))


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(video_rows: list[dict], selected_rows: list[dict], output_dir: Path) -> None:
    lines = [
        "# Selected BO Advantage 3D Videos",
        "",
        "| layout | run | AGV | case | policy | total_time | file |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['layout_key']} | {row['run_index']} | {row['total_agvs']} | "
            f"{row['scenario_case_id']} | {row['policy']} | {row['total_time']} | "
            f"{Path(row['video_path']).name} |"
        )
    lines.extend(["", "## Selection Rows", ""])
    for row in selected_rows:
        lines.append(
            f"- {row['layout_key']}: run {row['run_index']}, "
            f"BO margin vs best other = {row['bo_margin_vs_best_other']}"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
