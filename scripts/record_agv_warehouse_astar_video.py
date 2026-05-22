"""Record the warehouse AGV A* simulation to MP4 or GIF."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from rware.agv_layouts import build_warehouse_aisle_layout_v1
from rware.agv_simulation_wrapper import AGVRWARESimulationWrapper
from scripts.run_agv_intersection_demo import (
    assign_initial_missions,
    scripted_actions,
    update_missions,
)


CELL_SIZE = 28
PAD_TOP = 70
PAD_LEFT = 16
PANEL_WIDTH = 360

COLORS = {
    "background": (245, 246, 248),
    "shelf": (84, 72, 137),
    "road": (255, 255, 255),
    "grid": (220, 224, 230),
    "station": (166, 222, 255),
    "task": (40, 40, 220),
    "waiting": (40, 170, 240),
    "conflict": (70, 70, 210),
    "text": (35, 35, 35),
}

ROBOT_COLORS = {
    "AGV_1": (0, 140, 255),
    "AGV_2": (40, 170, 40),
    "AGV_3": (210, 80, 170),
    "AGV_4": (30, 180, 220),
}


def record_video(output_dir: Path, steps: int, fps: int, video_format: str) -> Path:
    """Record the A* warehouse simulation.

    Args:
        output_dir: Directory where the video is written.
        steps: Number of simulation steps to record.
        fps: Output video FPS.
        video_format: mp4 or gif.

    Returns:
        Path to the written video file.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    layout = build_warehouse_aisle_layout_v1()
    wrapper = AGVRWARESimulationWrapper(layout, max_steps=steps)
    wrapper.reset(seed=0)
    assign_initial_missions(wrapper)

    frames = [_draw_frame(wrapper, None)]
    for _ in range(steps):
        update_missions(wrapper)
        result = wrapper.step(scripted_actions(wrapper))
        frames.append(_draw_frame(wrapper, result))

    path = output_dir / f"agv_warehouse_astar_steps{steps}.{video_format}"
    _write_video(path, frames, fps, video_format)
    wrapper.close()
    return path


def _draw_frame(wrapper: AGVRWARESimulationWrapper, step_result: dict | None) -> np.ndarray:
    layout = wrapper.layout
    rows, cols = layout.grid_size
    width = PAD_LEFT * 2 + cols * CELL_SIZE + PANEL_WIDTH
    height = PAD_TOP + rows * CELL_SIZE + 18
    frame = np.full((height, width, 3), COLORS["background"], dtype=np.uint8)

    _draw_title(frame, wrapper, step_result)
    _draw_layout(frame, wrapper)
    _draw_interaction_areas(frame, wrapper)
    _draw_tasks_and_stations(frame, wrapper)
    _draw_agvs(frame, wrapper)
    _draw_panel(frame, wrapper, step_result)
    return frame


def _draw_title(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper, step_result: dict | None) -> None:
    step = wrapper.simulator.step_count if step_result is None else step_result["step"]
    cv2.putText(
        frame,
        f"Warehouse AGV A* Demo | step {step}",
        (PAD_LEFT, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLORS["text"],
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "G = station / delivery-unload goal, purple = shelf storage, red ring = assigned task",
        (PAD_LEFT, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLORS["text"],
        1,
        cv2.LINE_AA,
    )


def _draw_layout(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper) -> None:
    road_positions = {node.position for node in wrapper.layout.graph.nodes.values()}
    station_positions = {
        wrapper.layout.graph.nodes[node_id].position for node_id in wrapper.layout.stations.values()
    }
    rows, cols = wrapper.layout.grid_size
    for row in range(rows):
        for col in range(cols):
            color = COLORS["road"] if (row, col) in road_positions else COLORS["shelf"]
            if (row, col) in station_positions:
                color = COLORS["station"]
            x0, y0 = _cell_top_left(row, col)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL_SIZE, y0 + CELL_SIZE), color, -1)
            cv2.rectangle(frame, (x0, y0), (x0 + CELL_SIZE, y0 + CELL_SIZE), COLORS["grid"], 1)


def _draw_interaction_areas(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper) -> None:
    graph = wrapper.layout.graph
    for area in wrapper.layout.interaction_areas:
        for node_id in area.conflict_points:
            if node_id in graph.nodes:
                _draw_node_marker(frame, graph.nodes[node_id].position, COLORS["conflict"], "C")
        for node_id in area.waiting_points:
            if node_id in graph.nodes:
                _draw_node_marker(frame, graph.nodes[node_id].position, COLORS["waiting"], "W")


def _draw_tasks_and_stations(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper) -> None:
    graph = wrapper.layout.graph
    for station_name, node_id in wrapper.layout.stations.items():
        row, col = graph.nodes[node_id].position
        x, y = _cell_center(row, col)
        cv2.putText(frame, "G", (x - 9, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 80, 120), 2, cv2.LINE_AA)
    for task_name, node_id in wrapper.layout.task_nodes.items():
        row, col = graph.nodes[node_id].position
        x, y = _cell_center(row, col)
        cv2.circle(frame, (x, y), 8, COLORS["task"], 2, cv2.LINE_AA)


def _draw_agvs(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper) -> None:
    graph = wrapper.layout.graph
    for agv in wrapper.simulator.agvs:
        row, col = graph.nodes[agv.current_node].position
        x, y = _cell_center(row, col)
        color = ROBOT_COLORS.get(agv.robot_id, (0, 0, 0))
        cv2.circle(frame, (x, y), 10, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 11, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, agv.robot_id[-1], (x - 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        dx, dy = _heading_delta(agv.heading.name)
        cv2.arrowedLine(frame, (x, y), (x + dx * 16, y + dy * 16), color, 2, cv2.LINE_AA, tipLength=0.35)


def _draw_panel(frame: np.ndarray, wrapper: AGVRWARESimulationWrapper, step_result: dict | None) -> None:
    x0 = PAD_LEFT + wrapper.layout.grid_size[1] * CELL_SIZE + 24
    y = PAD_TOP
    lines = [
        "Policy: A* route assignment",
        f"Robots: {len(wrapper.simulator.agvs)}",
        "Purpose: G -> task -> G",
        "",
        "AGV missions",
    ]
    for agv in wrapper.simulator.agvs:
        mission = agv.mission
        phase = mission.phase if mission else "none"
        pickup = mission.pickup_node if mission else "-"
        dropoff = mission.dropoff_node if mission else "-"
        lines.append(f"{agv.robot_id}: {phase}")
        lines.append(f"  {pickup} -> {dropoff}")
    lines.extend(["", "Recent conflict events"])
    events = wrapper.simulator.collision_events[-5:]
    if events:
        for event in events:
            lines.append(f"s{event['step']} {event['event_type']}")
            lines.append(f"  robots={event['robots']} node={event['node']}")
    else:
        lines.append("none")

    for line in lines:
        cv2.putText(frame, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, COLORS["text"], 1, cv2.LINE_AA)
        y += 20


def _draw_node_marker(frame: np.ndarray, position: tuple[int, int], color: tuple[int, int, int], label: str) -> None:
    x, y = _cell_center(*position)
    cv2.circle(frame, (x, y), 7, color, -1, cv2.LINE_AA)
    cv2.putText(frame, label, (x - 4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)


def _cell_top_left(row: int, col: int) -> tuple[int, int]:
    return PAD_LEFT + col * CELL_SIZE, PAD_TOP + row * CELL_SIZE


def _cell_center(row: int, col: int) -> tuple[int, int]:
    x0, y0 = _cell_top_left(row, col)
    return x0 + CELL_SIZE // 2, y0 + CELL_SIZE // 2


def _heading_delta(heading: str) -> tuple[int, int]:
    return {
        "NORTH": (0, -1),
        "EAST": (1, 0),
        "SOUTH": (0, 1),
        "WEST": (-1, 0),
    }[heading]


def _write_video(path: Path, frames: list[np.ndarray], fps: int, video_format: str) -> None:
    if video_format == "gif":
        imageio.mimsave(path, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], duration=1 / max(1, fps))
        return

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], duration=1 / max(1, fps))
        return
    for frame in frames:
        writer.write(frame)
    writer.release()


def main() -> None:
    """Parse CLI arguments and record the video."""

    parser = argparse.ArgumentParser(description="Record warehouse AGV A* simulation video.")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "agv_warehouse_astar_video"))
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    args = parser.parse_args()

    path = record_video(Path(args.output_dir), args.steps, args.fps, args.format)
    print(path)


if __name__ == "__main__":
    main()
