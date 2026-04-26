import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import rware

from deadlock.dsd_fsd.experiment import (
    _build_points,
    _make_controller,
    _normalise_intentional_waits,
    _record_health_metrics,
    _sample_kind,
    _sample_target,
    _stage_agents,
)
from deadlock.dsd_fsd.controllers import (
    DeadlockProneRuleController,
    LocalGraphAdmissionController,
)
from deadlock.dsd_fsd.layouts import (
    buffer_for_target,
    build_paper_points,
    build_shared_area_points,
    zone_for_target,
)
from deadlock.dsd_fsd.metrics import summarize_metrics
from deadlock.dsd_fsd.model import SystemType, TransactionLedger
from deadlock.dsd_fsd.shared_area_plans import (
    deadlock_plan,
    seed_deadlock_transactions,
)


BASELINE_SCENARIOS = [
    {
        "name": "01_baseline_uniform_dsd",
        "system": SystemType.DSD,
        "layout": "baseline",
        "workload": "uniform",
        "agents": 3,
        "stall_threshold": 6,
    },
    {
        "name": "01_baseline_uniform_fsd",
        "system": SystemType.FSD,
        "layout": "baseline",
        "workload": "uniform",
        "agents": 3,
        "stall_threshold": 6,
    },
]


SHARED_AREA_SCENARIOS = [
    {
        "name": "02_shared_cross_deadlock_dsd",
        "system": SystemType.DSD,
        "layout": "shared_area",
        "workload": "uniform",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_cross",
    },
    {
        "name": "02_shared_cross_deadlock_fsd",
        "system": SystemType.FSD,
        "layout": "shared_area",
        "workload": "uniform",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_cross",
    },
    {
        "name": "03_shared_queue_deadlock_dsd",
        "system": SystemType.DSD,
        "layout": "shared_area",
        "workload": "hotspot",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_queue",
    },
    {
        "name": "03_shared_queue_deadlock_fsd",
        "system": SystemType.FSD,
        "layout": "shared_area",
        "workload": "hotspot",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_queue",
    },
    {
        "name": "04_shared_cross_gnn_admission",
        "system": SystemType.FSD,
        "layout": "shared_area",
        "workload": "uniform",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_cross",
        "controller": "gnn_admission",
    },
    {
        "name": "05_shared_queue_gnn_admission",
        "system": SystemType.FSD,
        "layout": "shared_area",
        "workload": "hotspot",
        "agents": 4,
        "stall_threshold": 8,
        "deadlock_pattern": "shared_queue",
        "controller": "gnn_admission",
    },
]


SCENARIOS = [*BASELINE_SCENARIOS, *SHARED_AREA_SCENARIOS]


def record_scenario(
    scenario,
    output_dir,
    seed=0,
    steps=1000,
    arrival_interval=5,
    service_steps=3,
    fps=12,
    frame_stride=2,
    video_format="mp4",
):
    rng = np.random.default_rng(seed)
    system = scenario["system"]
    points = _build_video_points(scenario, system)
    plan = deadlock_plan(scenario, points)
    env = gym.make(
        f"rware:rware-tiny-{scenario['agents']}ag-v2",
        layout=points.layout,
        render_mode="rgb_array",
        max_steps=steps + 5,
        request_queue_size=0,
    )
    env.reset(seed=seed)
    _stage_agents(
        env,
        points,
        scenario["agents"],
        agent_starts=plan["starts"] if plan else None,
    )

    ledger = TransactionLedger()
    controller = _make_video_controller(
        scenario,
        system,
        env,
        points,
        service_steps,
        scenario["stall_threshold"],
    )
    if plan:
        seed_deadlock_transactions(ledger, plan)

    frames = [env.render()]
    for step in range(steps):
        if not plan and step % arrival_interval == 0:
            target = _sample_target(rng, points, scenario["workload"])
            ledger.create(
                target=target,
                zone_id=zone_for_target(points, target),
                step=step,
                kind=_sample_kind(rng),
                buffer=buffer_for_target(points, target),
            )

        actions = controller.compute_actions(step, ledger)
        _, _, done, truncated, _ = env.step(actions)
        _record_health_metrics(env, controller)
        _normalise_intentional_waits(env, controller)

        if step % frame_stride == 0:
            frames.append(env.render())
        if done or truncated:
            break

    controller.compute_actions(steps, ledger)
    metrics = summarize_metrics(ledger, controller.stats, steps)
    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{scenario['name']}.{video_format}"
    _write_video(video_path, frames, fps=fps, video_format=video_format)
    return video_path, metrics


def _build_video_points(scenario, system):
    if scenario["layout"] == "shared_area":
        return build_shared_area_points(n_agents=scenario["agents"])
    if scenario["layout"] == "paper" and "bay_rows" in scenario:
        return build_paper_points(
            system=system.value,
            n_agents=scenario["agents"],
            bay_rows=scenario["bay_rows"],
        )
    return _build_points(scenario["layout"], system, scenario["agents"])


def _make_video_controller(scenario, system, env, points, service_steps, stall_threshold):
    if scenario.get("controller") == "gnn_admission":
        return LocalGraphAdmissionController(env, points, service_steps=service_steps)
    if scenario.get("deadlock_pattern"):
        return DeadlockProneRuleController(env, points, service_steps=service_steps)
    return _make_controller(system, env, points, service_steps, stall_threshold)


def _write_video(path, frames, fps=12, video_format="mp4"):
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


def _pad_even(frame):
    height, width = frame.shape[:2]
    pad_height = height % 2
    pad_width = width % 2
    if not pad_height and not pad_width:
        return frame
    return np.pad(
        frame,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode="edge",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Record shared-area AMR deadlock/admission videos.")
    parser.add_argument("--output-dir", default=str(Path.home() / "Desktop" / "shared_area_videos"))
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Also record baseline DSD/FSD videos. By default only shared-area research scenarios are recorded.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--arrival-interval", type=int, default=5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    scenarios = SCENARIOS if args.include_baseline else SHARED_AREA_SCENARIOS
    for scenario in scenarios:
        path, metrics = record_scenario(
            scenario,
            output_dir,
            seed=args.seed,
            steps=args.steps,
            arrival_interval=args.arrival_interval,
            fps=args.fps,
            frame_stride=args.frame_stride,
            video_format=args.format,
        )
        print(
            f"{path} | completed={metrics['completed']} "
            f"f_avg={metrics['f_avg']:.2f} f_max={metrics['f_max']} "
            f"max_active_wait={metrics.get('max_active_wait', 0)}"
        )


if __name__ == "__main__":
    main()
