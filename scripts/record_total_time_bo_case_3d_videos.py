"""Record smooth isometric 3D videos for one total-time comparison case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment
from scripts.run_total_time_bo_policy_comparison import FIXED_HEURISTIC_WEIGHTS


POLICY_LABELS = {
    "fcfs": "FCFS",
    "fixed_heuristic": "Basic Heuristic",
    "bo_heuristic": "BO Heuristic",
}
TEAM_LABEL = "26-1 JS 1Team"
RIGHT_HAND_CONFLICT_EDGES = {
    ("CP_NW", "CP_SW"),
    ("CP_SW", "CP_SE"),
    ("CP_SE", "CP_NE"),
    ("CP_NE", "CP_NW"),
}
COLORS = {
    "background": (213, 215, 214),
    "floor_dark": (191, 194, 193),
    "floor_line": (177, 180, 178),
    "road": (226, 228, 226),
    "road_edge": (152, 158, 160),
    "road_side": (180, 184, 183),
    "lane_mark": (244, 246, 240),
    "waiting": (92, 159, 178),
    "conflict": (80, 99, 129),
    "start": (118, 159, 128),
    "exit": (123, 111, 98),
    "text": (31, 34, 38),
    "muted": (78, 84, 89),
    "shadow": (38, 43, 47),
    "metal": (58, 65, 72),
    "tire": (22, 25, 28),
    "headlight": (236, 244, 210),
}
ROBOT_COLORS = {
    "NORTH": (42, 114, 184),
    "SOUTH": (66, 142, 74),
    "WEST": (172, 80, 112),
    "EAST": (196, 118, 48),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record smooth 3D videos for FCFS, Basic Heuristic, and BO Heuristic."
    )
    parser.add_argument(
        "--experiment-dir",
        default=str(Path("final_output") / "agv24_policy_comparison"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path("final_output") / "agv24_policy_comparison" / "videos_3d_smooth"),
    )
    parser.add_argument("--run-index", type=int, default=-1)
    parser.add_argument(
        "--selection",
        choices=["bo_better", "max_spread"],
        default="bo_better",
    )
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--admission-window-steps", type=int, default=2)
    parser.add_argument("--shared-area-capacity", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames-per-step", type=int, default=8)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--allow-route-violations", action="store_true")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = _read_csv(experiment_dir / "experiment_settings.csv")
    comparisons = _read_csv(experiment_dir / "trial_comparison.csv")
    weight_metadata = json.loads(
        (experiment_dir / "bo_best_weights.json").read_text(encoding="utf-8")
    )
    fixed_weights = weight_metadata.get(
        "fixed_greedy_best_weights",
        weight_metadata.get("fixed_heuristic_weights", FIXED_HEURISTIC_WEIGHTS),
    )
    bo_weights = weight_metadata["best_bo_weights"]
    shared_area_capacity = (
        args.shared_area_capacity
        if args.shared_area_capacity is not None
        else int(weight_metadata.get("shared_area_capacity", 2))
    )

    selected = _select_case(comparisons, args.run_index, args.selection)
    setting = next(row for row in settings if int(row["run_index"]) == int(selected["run_index"]))
    scenario = _scenario_from_row(setting)
    route_violations = _route_violations(
        scenario=scenario,
        corridor_length=args.corridor_length,
        west_exit_extension=args.west_exit_extension,
        admission_window_steps=args.admission_window_steps,
        shared_area_capacity=shared_area_capacity,
    )
    if route_violations and not args.allow_route_violations:
        raise RuntimeError(
            "Right-hand shared-zone route validation failed: "
            + json.dumps(route_violations, sort_keys=True)
        )

    video_rows = []
    for policy in ("fcfs", "fixed_heuristic", "bo_heuristic"):
        path, metrics = _record_policy_3d_video(
            policy=policy,
            scenario=scenario,
            fixed_weights=fixed_weights,
            bo_weights=bo_weights,
            max_steps=args.max_steps,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            admission_window_steps=args.admission_window_steps,
            shared_area_capacity=shared_area_capacity,
            fps=args.fps,
            frames_per_step=args.frames_per_step,
            width=args.width,
            height=args.height,
            output_dir=output_dir,
        )
        video_rows.append(
            {
                "run_index": selected["run_index"],
                "policy": policy,
                "total_time": metrics["total_time"],
                "completed": metrics["completed"],
                "video_path": str(path),
            }
        )
        print(f"{policy}: {path} total_time={metrics['total_time']}")

    _write_csv(video_rows, output_dir / "selected_3d_video_case.csv")
    _write_readme(
        output_dir=output_dir,
        selected=selected,
        setting=setting,
        video_rows=video_rows,
        fixed_weights=fixed_weights,
        bo_weights=bo_weights,
        shared_area_capacity=shared_area_capacity,
        route_violations=route_violations,
    )
    print(f"selected run_index={selected['run_index']}")
    print(f"videos: {output_dir.resolve()}")


def _record_policy_3d_video(
    policy: str,
    scenario: dict,
    fixed_weights: dict[str, float],
    bo_weights: dict[str, float],
    max_steps: int,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
    fps: int,
    frames_per_step: int,
    width: int,
    height: int,
    output_dir: Path,
) -> tuple[Path, dict]:
    experiment = _build_experiment(
        policy=policy,
        scenario=scenario,
        fixed_weights=fixed_weights,
        bo_weights=bo_weights,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        admission_window_steps=admission_window_steps,
        shared_area_capacity=shared_area_capacity,
    )
    snapshots = [_snapshot(experiment)]
    while not experiment.is_done and experiment.step_count < max_steps:
        experiment.step()
        snapshots.append(_snapshot(experiment))

    metrics = experiment.metrics()
    path = output_dir / f"run{scenario['run_index']:03d}_{policy}_3d_total{metrics['total_time']}.mp4"
    _write_smooth_3d_video(
        path=path,
        snapshots=snapshots,
        experiment=experiment,
        policy=policy,
        metrics=metrics,
        fps=fps,
        frames_per_step=frames_per_step,
        width=width,
        height=height,
    )
    return path, metrics


def _build_experiment(
    policy: str,
    scenario: dict,
    fixed_weights: dict[str, float],
    bo_weights: dict[str, float],
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> FCFSCrossExperiment:
    if policy == "fcfs":
        policy_type = "fcfs"
        weights = None
    elif policy == "fixed_heuristic":
        policy_type = "heuristic"
        weights = fixed_weights
    else:
        policy_type = "heuristic"
        weights = bo_weights
    return FCFSCrossExperiment(
        robots_by_direction=scenario["allocation"],
        random_seed=scenario["seed"],
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=0,
        spawn_plan_by_direction=scenario["spawn_plan"],
        admission_window_steps=admission_window_steps,
        shared_area_capacity=shared_area_capacity,
        normalize_heuristic_features=True,
        goal_plan_by_direction=scenario["goal_plan"],
        policy_type=policy_type,
        heuristic_weights=weights,
    )


def _snapshot(experiment: FCFSCrossExperiment) -> dict:
    robots = {}
    for robot_id, robot in experiment.active.items():
        robots[robot_id] = {
            "node": robot.current_node,
            "direction": robot.direction,
            "status": robot.status,
        }
    for robot in experiment.completed:
        robots[robot.robot_id] = {
            "node": robot.goal_node,
            "direction": robot.direction,
            "status": "completed",
        }
    return {
        "step": experiment.step_count,
        "robots": robots,
        "shared_robot_ids": sorted(experiment.shared_robot_ids),
        "queue": list(experiment.fcfs_queue),
    }


def _write_smooth_3d_video(
    path: Path,
    snapshots: list[dict],
    experiment: FCFSCrossExperiment,
    policy: str,
    metrics: dict,
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
            renderer.draw_robots(frame, left, right, alpha)
            renderer.draw_overlay(frame, policy, metrics, left, alpha)
            writer.write(frame)
    frame = base.copy()
    renderer.draw_robots(frame, snapshots[-1], snapshots[-1], 1.0)
    renderer.draw_overlay(frame, policy, metrics, snapshots[-1], 1.0)
    writer.write(frame)
    writer.release()


class IsoRenderer:
    def __init__(self, experiment: FCFSCrossExperiment, width: int, height: int) -> None:
        self.experiment = experiment
        self.width = width
        self.height = height
        self.scale_x = 42.0
        self.scale_y = 23.0
        self.scale_z = 76.0
        self.origin_x = width / 2
        self.origin_y = 86.0
        self.road_positions = {
            node.position for node in experiment.graph.nodes.values()
        }
        self.start_nodes = set(experiment.START_BY_DIRECTION.values())
        self.exit_nodes = set(experiment.EXIT_BY_DIRECTION.values())

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
            elif node_id in self.experiment.area.conflict_zone_nodes:
                color = COLORS["conflict"]
                label = "C"
            elif node_id in self.experiment.area.waiting_points:
                color = COLORS["waiting"]
                label = "W"
            self.draw_tile(frame, row, col, color, label)
        self.draw_shared_zone_outline(frame)
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
            cv2.line(
                frame,
                (offset, 0),
                (offset + self.height, self.height),
                COLORS["floor_line"],
                1,
                cv2.LINE_AA,
            )
            cv2.line(
                frame,
                (offset, self.height),
                (offset + self.height, 0),
                COLORS["floor_line"],
                1,
                cv2.LINE_AA,
            )

    def draw_tile(
        self,
        frame: np.ndarray,
        row: float,
        col: float,
        color: tuple[int, int, int],
        label: str,
    ) -> None:
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
            cv2.putText(
                frame,
                label,
                (int(x) - 7, int(y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                text_color,
                1,
                cv2.LINE_AA,
            )

    def draw_lane_mark(self, frame: np.ndarray, row: float, col: float) -> None:
        node_id = self.experiment.graph.position_to_node.get((int(row), int(col)))
        if node_id in self.experiment.area.conflict_zone_nodes:
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

    def draw_shared_zone_outline(self, frame: np.ndarray) -> None:
        conflict_positions = [
            self.experiment.graph.nodes[node_id].position
            for node_id in self.experiment.area.conflict_zone_nodes
        ]
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

    def draw_robots(
        self,
        frame: np.ndarray,
        left: dict,
        right: dict,
        alpha: float,
    ) -> None:
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
            drow = right_row - left_row
            dcol = right_col - left_col
            placements.append((row + col, robot_id, row, col, drow, dcol, state))
        for _, robot_id, row, col, drow, dcol, state in sorted(placements):
            self.draw_agv(frame, row=row, col=col, drow=drow, dcol=dcol, robot_id=robot_id, state=state)

    def draw_agv(
        self,
        frame: np.ndarray,
        row: float,
        col: float,
        drow: float,
        dcol: float,
        robot_id: str,
        state: dict,
    ) -> None:
        color = ROBOT_COLORS[state["direction"]]
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
        right_face = np.array(
            [
                self.project(x1, y0, z1),
                self.project(x1, y1, z1),
                self.project(x1, y1, z0),
                self.project(x1, y0, z0),
            ],
            dtype=np.int32,
        )
        front_face = np.array(
            [
                self.project(x0, y1, z1),
                self.project(x1, y1, z1),
                self.project(x1, y1, z0),
                self.project(x0, y1, z0),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, front_face, _blend(color, (0, 0, 0), 0.28), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, right_face, _blend(color, (0, 0, 0), 0.42), cv2.LINE_AA)
        cv2.fillConvexPoly(frame, top, color, cv2.LINE_AA)
        cv2.polylines(frame, [top, right_face, front_face], True, (230, 235, 238), 1, cv2.LINE_AA)
        self.draw_agv_details(frame, row, col, x0, x1, y0, y1, z1, drow, dcol, color)
        cx, cy = self.project(col, row, z1 + 0.02)
        label = _robot_number(robot_id)
        cv2.putText(
            frame,
            label,
            (int(cx) - 6 * len(label), int(cy) + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def draw_agv_details(
        self,
        frame: np.ndarray,
        row: float,
        col: float,
        x0: float,
        x1: float,
        y0: float,
        y1: float,
        z1: float,
        drow: float,
        dcol: float,
        color: tuple[int, int, int],
    ) -> None:
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
            drow, dcol = {
                "NORTH": (-1.0, 0.0),
                "SOUTH": (1.0, 0.0),
                "WEST": (0.0, -1.0),
                "EAST": (0.0, 1.0),
            }.get("EAST", (0.0, 1.0))
        norm = math.hypot(drow, dcol) or 1.0
        front_row = row + 0.31 * drow / norm
        front_col = col + 0.31 * dcol / norm
        hx, hy = self.project(front_col, front_row, z1 + 0.035)
        cv2.circle(frame, (hx, hy), 3, COLORS["headlight"], -1, cv2.LINE_AA)
        cv2.circle(frame, (hx, hy), 5, _blend(COLORS["headlight"], color, 0.62), 1, cv2.LINE_AA)

        wheel_z = 0.16
        wheel_points = [
            (x0, (y0 + y1) / 2),
            (x1, (y0 + y1) / 2),
        ] if abs(dcol) >= abs(drow) else [
            ((x0 + x1) / 2, y0),
            ((x0 + x1) / 2, y1),
        ]
        for wheel_col, wheel_row in wheel_points:
            wx, wy = self.project(wheel_col, wheel_row, wheel_z)
            cv2.ellipse(frame, (wx, wy), (7, 4), 0, 0, 360, COLORS["tire"], -1, cv2.LINE_AA)
            cv2.ellipse(frame, (wx, wy), (4, 2), 0, 0, 360, COLORS["metal"], -1, cv2.LINE_AA)

    def draw_overlay(self, frame: np.ndarray, policy: str, metrics: dict, snapshot: dict, alpha: float) -> None:
        step = snapshot["step"] + alpha
        title = f"{POLICY_LABELS[policy]} | {TEAM_LABEL} | AGV {metrics['robots']} | step {step:.1f}"
        cv2.putText(frame, title, (32, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.78, COLORS["text"], 2, cv2.LINE_AA)
        subtitle = (
            f"A* right-hand route, shared capacity={metrics['shared_area_capacity']}, "
            f"total_time={metrics['total_time']}"
        )
        cv2.putText(frame, subtitle, (32, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLORS["muted"], 1, cv2.LINE_AA)

    def project(self, x: float, y: float, z: float) -> tuple[int, int]:
        return (
            int(round(self.origin_x + (x - y) * self.scale_x)),
            int(round(self.origin_y + (x + y) * self.scale_y - z * self.scale_z)),
        )


def _route_violations(
    scenario: dict,
    corridor_length: int,
    west_exit_extension: int,
    admission_window_steps: int,
    shared_area_capacity: int,
) -> list[dict]:
    experiment = _build_experiment(
        policy="fcfs",
        scenario=scenario,
        fixed_weights={},
        bo_weights={},
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        admission_window_steps=admission_window_steps,
        shared_area_capacity=shared_area_capacity,
    )
    robots = list(experiment.active.values())
    for queue in experiment.pending_by_direction.values():
        robots.extend(queue)
    violations = []
    for robot in robots:
        conflicts = [
            node_id
            for node_id in robot.path
            if node_id in experiment.area.conflict_zone_nodes
        ]
        for from_node, to_node in zip(conflicts, conflicts[1:]):
            if (from_node, to_node) not in RIGHT_HAND_CONFLICT_EDGES:
                violations.append(
                    {
                        "robot_id": robot.robot_id,
                        "route": robot.route_id,
                        "from": from_node,
                        "to": to_node,
                        "conflict_sequence": conflicts,
                    }
                )
    return violations


def _select_case(rows: list[dict], run_index: int, selection: str) -> dict:
    if run_index >= 0:
        return next(row for row in rows if int(row["run_index"]) == run_index)
    if selection == "bo_better":
        candidates = [
            row
            for row in rows
            if int(row["fcfs_total_time"]) > int(row["bo_heuristic_total_time"])
            and int(row["fixed_heuristic_total_time"]) > int(row["bo_heuristic_total_time"])
        ]
        if candidates:
            return max(
                candidates,
                key=lambda row: (
                    int(row["fcfs_total_time"]) - int(row["bo_heuristic_total_time"]),
                    int(row["fixed_heuristic_total_time"]) - int(row["bo_heuristic_total_time"]),
                ),
            )
    return max(
        rows,
        key=lambda row: _spread(
            int(row["fcfs_total_time"]),
            int(row["fixed_heuristic_total_time"]),
            int(row["bo_heuristic_total_time"]),
        ),
    )


def _spread(*values: int) -> int:
    return max(values) - min(values)


def _scenario_from_row(row: dict) -> dict:
    return {
        "run_index": int(row["run_index"]),
        "seed": int(row["scenario_seed"]),
        "allocation": {
            "NORTH": int(row["north_agvs"]),
            "SOUTH": int(row["south_agvs"]),
            "WEST": int(row["west_agvs"]),
            "EAST": int(row["east_agvs"]),
        },
        "spawn_plan": json.loads(row["spawn_plan_json"]),
        "goal_plan": json.loads(row["goal_plan_json"]),
    }


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    output_dir: Path,
    selected: dict,
    setting: dict,
    video_rows: list[dict],
    fixed_weights: dict,
    bo_weights: dict,
    shared_area_capacity: int,
    route_violations: list[dict],
) -> None:
    lines = [
        "# Smooth 3D Total-Time Video Case",
        "",
        f"- run_index: `{selected['run_index']}`",
        f"- allocation: NORTH={setting['north_agvs']}, SOUTH={setting['south_agvs']}, "
        f"WEST={setting['west_agvs']}, EAST={setting['east_agvs']}",
        f"- spawn_plan: `{setting['spawn_plan_json']}`",
        f"- goal_plan: `{setting['goal_plan_json']}`",
        f"- shared_area_capacity: `{shared_area_capacity}`",
        f"- right_hand_route_violations: `{len(route_violations)}`",
        f"- Fixed weights: `{json.dumps(fixed_weights, sort_keys=True)}`",
        f"- BO weights: `{json.dumps(bo_weights, sort_keys=True)}`",
        "",
        "| policy | total_time | video |",
        "|---|---:|---|",
    ]
    for row in video_rows:
        lines.append(
            f"| {row['policy']} | {row['total_time']} | `{Path(row['video_path']).name}` |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def _blend(left: tuple[int, int, int], right: tuple[int, int, int], ratio: float) -> tuple[int, int, int]:
    return tuple(
        int(round(left[index] * (1.0 - ratio) + right[index] * ratio))
        for index in range(3)
    )


def _robot_number(robot_id: str) -> str:
    token = robot_id.split("_", 1)[-1]
    return token[1:] if len(token) > 1 else token


if __name__ == "__main__":
    main()
