from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from deadlock.dsd_fsd.benchmark.config import ExperimentConfig
from deadlock.dsd_fsd.benchmark.layout_builder import build_layout
from deadlock.dsd_fsd.benchmark.simulator import WarehouseSimulator, generate_task_arrivals


POLICY_COLORS = {
    "rule_based": (245, 143, 24),
    "dsd": (76, 120, 168),
    "fsd": (84, 162, 75),
}
ZONE_COLORS = [
    (230, 242, 255),
    (255, 239, 219),
    (229, 246, 232),
    (245, 232, 246),
    (255, 247, 204),
]


def record_policy_video(
    config: ExperimentConfig,
    output_dir: Path,
    fps: int = 12,
    frame_stride: int = 2,
    cell_size: int = 24,
    video_format: str = "mp4",
) -> tuple[Path, dict]:
    """Record one policy simulation to MP4/GIF.

    Args:
        config: Experiment condition.
        output_dir: Directory where video is written.
        fps: Output frames per second.
        frame_stride: Record every N simulator steps.
        cell_size: Pixel size of one grid cell.
        video_format: mp4 or gif.

    Returns:
        Tuple of video path and final metrics.
    """

    layout = build_layout(config)
    arrivals = generate_task_arrivals(config, layout)
    simulator = WarehouseSimulator(config, layout=layout, task_arrivals=arrivals)
    frames = [render_frame(simulator, step=0, event=None, cell_size=cell_size)]

    for step in range(config.max_episode_steps):
        event = simulator.step(step)
        if step % frame_stride == 0:
            frames.append(render_frame(simulator, step=step + 1, event=event, cell_size=cell_size))

    metrics = simulator.metrics()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{config.policy_type}_{config.num_robots}robots_{config.num_aisles}aisles_"
        f"{config.capacity}_{config.arrival_rate}_seed{config.seed}.{video_format}"
    )
    path = output_dir / name
    _write_video(path, frames, fps=fps, video_format=video_format)
    return path, metrics


def record_comparison_videos(
    base_config: ExperimentConfig,
    policies: list[str],
    output_dir: Path,
    fps: int = 12,
    frame_stride: int = 2,
    cell_size: int = 24,
    video_format: str = "mp4",
) -> list[tuple[Path, dict]]:
    """Record policy videos under identical layout and demand sequence.

    Args:
        base_config: Shared condition. policy_type is replaced per policy.
        policies: Policy names to record.
        output_dir: Output directory.
        fps: Output frames per second.
        frame_stride: Record every N simulator steps.
        cell_size: Pixel size of one grid cell.
        video_format: mp4 or gif.

    Returns:
        List of video path and metrics pairs.
    """

    layout = build_layout(base_config)
    arrivals = generate_task_arrivals(base_config, layout)
    results = []
    for policy in policies:
        config = replace(base_config, policy_type=policy)
        simulator = WarehouseSimulator(config, layout=layout, task_arrivals=arrivals)
        frames = [render_frame(simulator, step=0, event=None, cell_size=cell_size)]
        for step in range(config.max_episode_steps):
            event = simulator.step(step)
            if step % frame_stride == 0:
                frames.append(render_frame(simulator, step=step + 1, event=event, cell_size=cell_size))
        metrics = simulator.metrics()
        output_dir.mkdir(parents=True, exist_ok=True)
        name = (
            f"{policy}_{config.num_robots}robots_{config.num_aisles}aisles_"
            f"{config.capacity}_{config.arrival_rate}_seed{config.seed}.{video_format}"
        )
        path = output_dir / name
        _write_video(path, frames, fps=fps, video_format=video_format)
        results.append((path, metrics))
    return results


def render_frame(simulator: WarehouseSimulator, step: int, event: dict | None, cell_size: int = 24) -> np.ndarray:
    """Draw the current simulator state into an RGB image.

    Args:
        simulator: Simulator to render.
        step: Displayed timestep.
        event: Optional event dictionary returned by simulator.step().
        cell_size: Pixel size of one grid cell.

    Returns:
        RGB image as a numpy array.
    """

    layout = simulator.layout
    sidebar_width = 300
    width = layout.width * cell_size + sidebar_width
    height = max(layout.height * cell_size, 360)
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    rows = layout.layout.splitlines()

    zone_by_cell = {}
    if simulator.config.policy_type == "dsd":
        for zone_id, zone in enumerate(layout.zones):
            for cell in zone:
                zone_by_cell[cell] = zone_id

    for y, row in enumerate(rows):
        for x, char in enumerate(row):
            cell = (x, y)
            color = _cell_color(simulator, char, cell, zone_by_cell)
            _fill_cell(frame, x, y, cell_size, color)
            _stroke_cell(frame, x, y, cell_size, (208, 208, 208))

    if simulator.config.policy_type == "fsd":
        for cell in layout.escape_points:
            _fill_role(frame, cell, cell_size, (116, 196, 118), "E")
        for cell in layout.waiting_points:
            _fill_role(frame, cell, cell_size, (121, 205, 205), "W")
        for cell in layout.decision_points:
            _fill_role(frame, cell, cell_size, (244, 203, 72), "D")

    waiting_tasks = [task for task in simulator.tasks.values() if task.is_waiting]
    active_tasks = [task for task in simulator.tasks.values() if task.assigned_robot_id is not None and not task.is_completed]
    for task in waiting_tasks:
        _draw_task(frame, task.target, cell_size, (210, 66, 66), fill=True)
    for task in active_tasks:
        _draw_task(frame, task.target, cell_size, (210, 66, 66), fill=False)

    for robot in simulator.robots:
        _draw_robot(frame, robot, simulator.config.policy_type, cell_size)

    _draw_sidebar(frame, simulator, step, event, layout.width * cell_size)
    return frame


def _cell_color(simulator: WarehouseSimulator, char: str, cell: tuple[int, int], zone_by_cell: dict) -> tuple[int, int, int]:
    if cell in zone_by_cell:
        return ZONE_COLORS[zone_by_cell[cell] % len(ZONE_COLORS)]
    if char == "x":
        return (78, 68, 138)
    if char == "g":
        return (120, 184, 232)
    if cell[1] in simulator.layout.transition_rows:
        return (238, 238, 238)
    return (248, 248, 248)


def _fill_cell(frame: np.ndarray, x: int, y: int, size: int, color: tuple[int, int, int]) -> None:
    x0, y0 = x * size, y * size
    frame[y0:y0 + size, x0:x0 + size] = color


def _stroke_cell(frame: np.ndarray, x: int, y: int, size: int, color: tuple[int, int, int]) -> None:
    x0, y0 = x * size, y * size
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), color, 1)


def _fill_role(frame: np.ndarray, cell: tuple[int, int], size: int, color: tuple[int, int, int], label: str) -> None:
    x, y = cell
    pad = max(2, size // 7)
    x0, y0 = x * size + pad, y * size + pad
    x1, y1 = (x + 1) * size - pad, (y + 1) * size - pad
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, -1)
    if size >= 20:
        cv2.putText(frame, label, (x0 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (35, 35, 35), 1, cv2.LINE_AA)


def _draw_task(frame: np.ndarray, cell: tuple[int, int], size: int, color: tuple[int, int, int], fill: bool) -> None:
    x, y = cell
    center = (x * size + size // 2, y * size + size // 2)
    radius = max(3, size // 5)
    cv2.circle(frame, center, radius, color, -1 if fill else 2, cv2.LINE_AA)


def _draw_robot(frame: np.ndarray, robot, policy_type: str, size: int) -> None:
    x, y = robot.position
    center = (x * size + size // 2, y * size + size // 2)
    color = POLICY_COLORS.get(policy_type, (240, 150, 30))
    if robot.wait_steps > 0:
        color = (215, 55, 55)
    radius = max(6, size // 3)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        str(robot.id),
        (center[0] - radius // 2, center[1] + radius // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_sidebar(frame: np.ndarray, simulator: WarehouseSimulator, step: int, event: dict | None, x0: int) -> None:
    cv2.rectangle(frame, (x0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (245, 245, 245), -1)
    metrics = simulator.metrics()
    blocked = list(event.get("blocked_ids", ())) if event else []
    lines = [
        f"Policy: {simulator.config.policy_type}",
        f"Step: {step}",
        f"Robots: {simulator.config.num_robots}",
        f"Aisles: {simulator.config.num_aisles}",
        f"Demand: {simulator.config.arrival_rate}",
        "",
        f"Throughput: {metrics['Throughput']}",
        f"Favg: {metrics['Favg']:.1f}",
        f"Fmax: {metrics['Fmax']}",
        f"Blocking: {metrics['BlockingTime']}",
        f"Deadlock: {metrics['DeadlockCount']}",
        f"Trigger: {metrics['TriggerCount']}",
        f"Reroute: {metrics['RerouteCount']}",
        f"Queue: {metrics['AvgQueueLength']:.2f}",
        "",
        f"Blocked now: {blocked}",
        "",
        "Legend:",
        "orange/blue/green = robot",
        "red robot = blocked/waiting",
        "red dot = waiting task",
        "red ring = assigned task",
        "D/W/E = FSD control points",
    ]
    y = 24
    for line in lines:
        cv2.putText(frame, line, (x0 + 14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y += 22


def _write_video(path: Path, frames: list[np.ndarray], fps: int = 12, video_format: str = "mp4") -> None:
    if video_format == "gif":
        imageio.mimsave(path, frames, duration=1 / max(1, fps))
        return

    first = _pad_even(frames[0])
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, duration=1 / max(1, fps))
        return

    for frame in frames:
        frame = _pad_even(frame)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _pad_even(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    pad_height = height % 2
    pad_width = width % 2
    if not pad_height and not pad_width:
        return frame
    return np.pad(frame, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record benchmark AMR simulation videos.")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "dsd_fsd_benchmark_videos"))
    parser.add_argument("--policies", default="rule_based,dsd,fsd")
    parser.add_argument("--robots", type=int, default=5)
    parser.add_argument("--aisles", type=int, default=6)
    parser.add_argument("--capacity", choices=["small", "large"], default="small")
    parser.add_argument("--arrival-rate", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--max-wait-steps", type=int, default=6)
    parser.add_argument("--deadlock-window", type=int, default=8)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--cell-size", type=int, default=24)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    base_config = ExperimentConfig(
        policy_type=policies[0] if policies else "rule_based",
        num_robots=args.robots,
        num_aisles=args.aisles,
        capacity=args.capacity,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
        max_episode_steps=args.steps,
        max_wait_steps=args.max_wait_steps,
        deadlock_window=args.deadlock_window,
    )
    results = record_comparison_videos(
        base_config,
        policies=policies,
        output_dir=Path(args.output_dir),
        fps=args.fps,
        frame_stride=args.frame_stride,
        cell_size=args.cell_size,
        video_format=args.format,
    )
    for path, metrics in results:
        print(
            f"{path} | throughput={metrics['Throughput']} "
            f"Favg={metrics['Favg']:.2f} Fmax={metrics['Fmax']} "
            f"blocking={metrics['BlockingTime']} deadlock={metrics['DeadlockCount']} "
            f"trigger={metrics['TriggerCount']} reroute={metrics['RerouteCount']}"
        )


if __name__ == "__main__":
    main()
