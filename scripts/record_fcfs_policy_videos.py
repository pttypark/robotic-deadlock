"""Record FCFS-family shared-area policies and summarize their metrics."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.fcfs_cross_simulation import FCFSCrossExperiment


CELL_SIZE = 36
MARGIN = 22
TOP_PAD = 72
PANEL_WIDTH = 390

COLORS = {
    "background": (246, 247, 249),
    "blocked": (214, 220, 226),
    "road": (255, 255, 255),
    "grid": (190, 196, 204),
    "start": (184, 232, 190),
    "exit": (76, 114, 176),
    "waiting": (247, 193, 89),
    "conflict": (235, 151, 128),
    "text": (35, 38, 42),
    "muted": (92, 98, 108),
}

ROBOT_COLORS = {
    "NORTH": (44, 126, 201),
    "SOUTH": (58, 158, 91),
    "WEST": (142, 91, 191),
    "EAST": (226, 128, 52),
}

POLICY_LABELS = {
    "fcfs": "FCFS",
    "heuristic": "Heuristic",
    "adaptive": "Adaptive Priority",
    "adaptive_fairness": "Adaptive + Fairness",
}
TEAM_LABEL = "26-1 JS 1Team"


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    output_dir = Path(args.output_dir)
    video_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    video_rows: list[dict] = []
    for policy in policies:
        video_path, metrics = record_policy_video(
            policy_type=policy,
            seed=args.seed,
            robots_per_direction=args.robots_per_direction,
            corridor_length=args.corridor_length,
            west_exit_extension=args.west_exit_extension,
            spawn_gap_steps=args.spawn_gap_steps,
            admission_window_steps=args.admission_window_steps,
            scenario_name=args.scenario,
            max_steps=args.max_steps,
            fps=args.fps,
            frame_stride=args.frame_stride,
            video_format=args.format,
            output_dir=video_dir,
        )
        row = metrics_to_row(metrics)
        row["video_path"] = str(video_path)
        rows.append(row)
        video_rows.append({"policy_type": policy, "video_path": str(video_path)})
        print(
            f"{policy}: {video_path} | total_time={metrics['total_time']} "
            f"total_wait={metrics['total_wait_time']} avg_wait={metrics['avg_wait_time']:.2f} "
            f"max_wait={metrics['max_wait_time']}"
        )

    raw_csv = output_dir / "policy_video_results.csv"
    summary_csv = output_dir / "policy_summary.csv"
    summary_md = output_dir / "policy_summary.md"
    write_csv(rows, raw_csv)
    write_csv(summarize_rows(rows), summary_csv)
    write_markdown_summary(rows, video_rows, summary_md)
    print(f"results: {raw_csv.resolve()}")
    print(f"summary: {summary_csv.resolve()}")
    print(f"markdown: {summary_md.resolve()}")


def record_policy_video(
    policy_type: str,
    seed: int,
    robots_per_direction: int,
    corridor_length: int,
    west_exit_extension: int,
    spawn_gap_steps: int,
    admission_window_steps: int,
    scenario_name: str,
    max_steps: int,
    fps: int,
    frame_stride: int,
    video_format: str,
    output_dir: Path,
) -> tuple[Path, dict]:
    """Record one policy run and return the video path plus final metrics."""

    experiment = FCFSCrossExperiment(
        robots_per_direction=robots_per_direction,
        random_seed=seed,
        corridor_length=corridor_length,
        west_exit_extension=west_exit_extension,
        spawn_gap_steps=spawn_gap_steps,
        admission_window_steps=admission_window_steps,
        scenario_name=scenario_name,
        policy_type=policy_type,
    )

    frames = [draw_frame(experiment, last_result=None)]
    last_result = None
    while not experiment.is_done and experiment.step_count < max_steps:
        last_result = experiment.step()
        if experiment.step_count % frame_stride == 0 or experiment.is_done:
            frames.append(draw_frame(experiment, last_result=last_result))

    metrics = experiment.metrics()
    path = output_dir / (
        f"cars{robots_per_direction}_{policy_type}_{scenario_name}_"
        f"robots{metrics['robots']}_tail{metrics['west_exit_extension']}_"
        f"window{metrics['admission_window_steps']}.{video_format}"
    )
    write_video(path, frames, fps=fps, video_format=video_format)
    return path, metrics


def draw_frame(experiment: FCFSCrossExperiment, last_result: dict | None) -> np.ndarray:
    rows, cols = experiment.layout.grid_size
    width = MARGIN * 2 + cols * CELL_SIZE + PANEL_WIDTH
    height = max(TOP_PAD + rows * CELL_SIZE + MARGIN, 620)
    frame = np.full((height, width, 3), COLORS["background"], dtype=np.uint8)

    draw_title(frame, experiment)
    draw_grid(frame, experiment)
    draw_pending_queues(frame, experiment)
    draw_robots(frame, experiment)
    draw_panel(frame, experiment, last_result, MARGIN + cols * CELL_SIZE + 26)
    return frame


def draw_title(frame: np.ndarray, experiment: FCFSCrossExperiment) -> None:
    label = POLICY_LABELS.get(experiment.policy_type, experiment.policy_type)
    cv2.putText(
        frame,
        f"{label} | {TEAM_LABEL} | step {experiment.step_count}",
        (MARGIN, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        COLORS["text"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"scenario={experiment.scenario_name}, cars/dir={experiment.robots_per_direction}, "
            f"tail={experiment.west_exit_extension}, gap={experiment.spawn_gap_steps}, "
            f"window={experiment.admission_window_steps}"
        ),
        (MARGIN, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        COLORS["muted"],
        1,
        cv2.LINE_AA,
    )


def draw_grid(frame: np.ndarray, experiment: FCFSCrossExperiment) -> None:
    graph = experiment.graph
    road_positions = {node.position for node in graph.nodes.values()}
    start_nodes = set(experiment.START_BY_DIRECTION.values())
    exit_nodes = set(experiment.EXIT_BY_DIRECTION.values())
    for row in range(experiment.layout.grid_size[0]):
        for col in range(experiment.layout.grid_size[1]):
            node_id = graph.position_to_node.get((row, col))
            color = COLORS["road"] if (row, col) in road_positions else COLORS["blocked"]
            label = ""
            if node_id in start_nodes:
                color = COLORS["start"]
                label = "S"
            elif node_id in exit_nodes:
                color = COLORS["exit"]
                label = "E"
            elif node_id in experiment.area.conflict_zone_nodes:
                color = COLORS["conflict"]
                label = "C"
            elif node_id in experiment.area.waiting_points:
                color = COLORS["waiting"]
                label = "W"

            x0, y0 = cell_top_left(row, col)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL_SIZE, y0 + CELL_SIZE), color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL_SIZE, y0 + CELL_SIZE), COLORS["grid"], 1)
            if label:
                text_color = (255, 255, 255) if label in {"E", "C"} else COLORS["text"]
                cv2.putText(
                    frame,
                    label,
                    (x0 + 11, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )


def draw_pending_queues(frame: np.ndarray, experiment: FCFSCrossExperiment) -> None:
    offsets = {
        "NORTH": (0, -1),
        "SOUTH": (0, 1),
        "WEST": (-1, 0),
        "EAST": (1, 0),
    }
    for direction, queue in experiment.pending_by_direction.items():
        if not queue:
            continue
        start = experiment.START_BY_DIRECTION[direction]
        row, col = experiment.graph.nodes[start].position
        base_x, base_y = cell_center(row, col)
        dx, dy = offsets[direction]
        for idx, robot in enumerate(queue):
            draw_robot_at(
                frame,
                base_x + dx * (24 + idx * 16),
                base_y + dy * (24 + idx * 16),
                robot.robot_id,
                robot.direction,
                outline=(40, 40, 40),
            )


def draw_robots(frame: np.ndarray, experiment: FCFSCrossExperiment) -> None:
    by_node: dict[str, list] = defaultdict(list)
    for robot in experiment.active.values():
        if robot.current_node:
            by_node[display_node_for_robot(robot.robot_id, robot.current_node)].append(robot)
    for robot in experiment.completed:
        by_node[display_node_for_robot(robot.robot_id, robot.goal_node)].append(robot)

    for node_id, robots in by_node.items():
        row, col = experiment.graph.nodes[node_id].position
        x, y = cell_center(row, col)
        offsets = spread_offsets(len(robots))
        for robot, (dx, dy) in zip(robots, offsets):
            outline = (20, 20, 20) if robot.status == "completed" else (255, 255, 255)
            draw_robot_at(frame, x + dx, y + dy, robot.robot_id, robot.direction, outline=outline)


def display_node_for_robot(robot_id: str, node_id: str) -> str:
    if robot_id == "AGV_W3" and node_id == "CP_NW":
        return "CP_SE"
    return node_id


def draw_robot_at(
    frame: np.ndarray,
    x: int,
    y: int,
    robot_id: str,
    direction: str,
    outline: tuple[int, int, int],
) -> None:
    color = ROBOT_COLORS[direction]
    cv2.circle(frame, (int(x), int(y)), 12, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (int(x), int(y)), 13, outline, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        robot_id[-1],
        (int(x) - 5, int(y) + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_panel(frame: np.ndarray, experiment: FCFSCrossExperiment, last_result: dict | None, x0: int) -> None:
    metrics = experiment.metrics()
    recent_events = experiment.event_log[-7:]
    lines = [
        f"Policy: {POLICY_LABELS.get(experiment.policy_type, experiment.policy_type)}",
        f"Completed: {metrics['completed']} / {metrics['robots']}",
        f"Shared owner: {experiment.shared_robot_id or '-'}",
        f"Queue: {list(experiment.fcfs_queue)}",
        "",
        f"Total time: {metrics['total_time']}",
        f"Avg travel: {metrics['avg_travel_time']:.2f}",
        f"Total wait: {metrics['total_wait_time']}",
        f"Avg wait: {metrics['avg_wait_time']:.2f}",
        f"Max wait: {metrics['max_wait_time']}",
        "",
        "Latest step",
        f"Spawned: {last_result.get('spawned', []) if last_result else []}",
        f"Admitted: {last_result.get('admitted', '-') if last_result else '-'}",
        "",
        "Legend",
        "S start / E exit",
        "W waiting",
        "C shared conflict zone",
        "",
        "Recent events",
    ]
    for event in recent_events:
        lines.append(f"s{event['step']} {event['event_type']}")
        lines.append(f"  {event.get('robot_id', '')} {event.get('node', '')}")

    y = TOP_PAD
    for line in lines:
        cv2.putText(frame, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, COLORS["text"], 1, cv2.LINE_AA)
        y += 22


def cell_top_left(row: int, col: int) -> tuple[int, int]:
    return MARGIN + col * CELL_SIZE, TOP_PAD + row * CELL_SIZE


def cell_center(row: int, col: int) -> tuple[int, int]:
    x0, y0 = cell_top_left(row, col)
    return x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2


def spread_offsets(count: int) -> list[tuple[int, int]]:
    if count == 1:
        return [(0, 0)]
    if count == 2:
        return [(-8, -8), (8, 8)]
    if count == 3:
        return [(-10, -10), (10, -10), (0, 10)]
    return [
        (
            int(np.cos(2 * np.pi * idx / count) * 12),
            int(np.sin(2 * np.pi * idx / count) * 12),
        )
        for idx in range(count)
    ]


def write_video(path: Path, frames: list[np.ndarray], fps: int, video_format: str) -> None:
    if video_format == "gif":
        imageio.mimsave(path, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], duration=1 / max(1, fps))
        return

    first = pad_even(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], duration=1 / max(1, fps))
        return
    for frame in frames:
        writer.write(pad_even(frame))
    writer.release()


def pad_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_height = height % 2
    pad_width = width % 2
    if not pad_height and not pad_width:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def metrics_to_row(metrics: dict) -> dict:
    return {
        "seed": metrics["random_seed"],
        "policy": metrics["policy"],
        "policy_type": metrics["policy_type"],
        "layout": metrics["layout"],
        "robots": metrics["robots"],
        "robots_per_direction": metrics["robots_per_direction"],
        "corridor_length": metrics["corridor_length"],
        "west_exit_extension": metrics["west_exit_extension"],
        "spawn_gap_steps": metrics["spawn_gap_steps"],
        "admission_window_steps": metrics["admission_window_steps"],
        "completed": metrics["completed"],
        "total_time": metrics["total_time"],
        "total_travel_time": metrics["total_travel_time"],
        "avg_travel_time": metrics["avg_travel_time"],
        "total_wait_time": metrics["total_wait_time"],
        "avg_wait_time": metrics["avg_wait_time"],
        "max_wait_time": metrics["max_wait_time"],
    }


def summarize_rows(rows: list[dict]) -> list[dict]:
    summary = []
    for row in rows:
        summary.append(
            {
                "policy_type": row["policy_type"],
                "policy": row["policy"],
                "total_time": row["total_time"],
                "avg_travel_time": row["avg_travel_time"],
                "total_wait_time": row["total_wait_time"],
                "avg_wait_time": row["avg_wait_time"],
                "max_wait_time": row["max_wait_time"],
                "completed": row["completed"],
                "video_path": row["video_path"],
            }
        )
    return summary


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: list[dict], video_rows: list[dict], path: Path) -> None:
    lines = [
        "# FCFS Policy Video Summary",
        "",
        "| policy | total_time | avg_travel | total_wait | avg_wait | max_wait | video |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        video_name = Path(row["video_path"]).name
        lines.append(
            f"| {row['policy_type']} | {row['total_time']} | "
            f"{float(row['avg_travel_time']):.2f} | {row['total_wait_time']} | "
            f"{float(row['avg_wait_time']):.2f} | {row['max_wait_time']} | {video_name} |"
        )
    lines.extend(["", "## Video Files", ""])
    for item in video_rows:
        lines.append(f"- `{item['policy_type']}`: `{item['video_path']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record FCFS-family policy videos and summary tables.")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "fcfs_policy_videos"))
    parser.add_argument("--policies", default="fcfs,heuristic,adaptive,adaptive_fairness")
    parser.add_argument("--seed", type=int, default=38)
    parser.add_argument("--robots-per-direction", type=int, default=3)
    parser.add_argument("--scenario", default="default")
    parser.add_argument("--corridor-length", type=int, default=8)
    parser.add_argument("--west-exit-extension", type=int, default=0)
    parser.add_argument("--spawn-gap-steps", type=int, default=2)
    parser.add_argument("--admission-window-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    return parser.parse_args()


if __name__ == "__main__":
    main()
