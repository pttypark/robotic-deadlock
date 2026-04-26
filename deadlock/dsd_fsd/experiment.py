import argparse
import json

import gymnasium as gym
import numpy as np
import rware
from rware.warehouse import Direction

from deadlock.dsd_fsd.controllers import DSDController, FSDTriggerController
from deadlock.dsd_fsd.layouts import (
    buffer_for_target,
    build_baseline_points,
    build_paper_points,
    zone_for_target,
)
from deadlock.dsd_fsd.metrics import summarize_metrics
from deadlock.dsd_fsd.model import SystemType, TransactionKind, TransactionLedger


def run_once(
    system=SystemType.FSD,
    seed=0,
    max_steps=200,
    n_agents=3,
    arrival_interval=5,
    service_steps=3,
    stall_threshold=8,
    layout_name="paper",
    workload="uniform",
    request_line_axis="aisle",
    request_line_index=None,
    agent_starts=None,
    render=False,
):
    system = SystemType(system)
    rng = np.random.default_rng(seed)
    points = _build_points(layout_name, system, n_agents)
    env = gym.make(
        f"rware:rware-tiny-{n_agents}ag-v2",
        layout=points.layout,
        render_mode="human" if render else "rgb_array",
        max_steps=max_steps + 5,
        request_queue_size=0,
    )
    env.reset(seed=seed)
    _stage_agents(env, points, n_agents, agent_starts=agent_starts)

    ledger = TransactionLedger()
    controller = _make_controller(system, env, points, service_steps, stall_threshold)

    for step in range(max_steps):
        if step % arrival_interval == 0:
            target = _sample_target(
                rng,
                points,
                workload,
                request_line_axis=request_line_axis,
                request_line_index=request_line_index,
            )
            kind = _sample_kind(rng)
            ledger.create(
                target=target,
                zone_id=zone_for_target(points, target),
                step=step,
                kind=kind,
                buffer=buffer_for_target(points, target),
            )

        actions = controller.compute_actions(step, ledger)
        _, _, done, truncated, _ = env.step(actions)
        _record_health_metrics(env, controller)
        _normalise_intentional_waits(env, controller)
        if render:
            env.render()
        if done or truncated:
            break

    controller.compute_actions(max_steps, ledger)
    env.close()
    return summarize_metrics(ledger, controller.stats, max_steps)


def run_comparison(
    seed=0,
    max_steps=200,
    n_agents=4,
    arrival_interval=5,
    stall_threshold=8,
    layout_name="paper",
    workload="uniform",
    request_line_axis="aisle",
    request_line_index=None,
    agent_starts=None,
):
    return {
        "dsd": run_once(
            SystemType.DSD,
            seed=seed,
            max_steps=max_steps,
            n_agents=n_agents,
            arrival_interval=arrival_interval,
            stall_threshold=stall_threshold,
            layout_name=layout_name,
            workload=workload,
            request_line_axis=request_line_axis,
            request_line_index=request_line_index,
            agent_starts=agent_starts,
        ),
        "fsd": run_once(
            SystemType.FSD,
            seed=seed,
            max_steps=max_steps,
            n_agents=n_agents,
            arrival_interval=arrival_interval,
            stall_threshold=stall_threshold,
            layout_name=layout_name,
            workload=workload,
            request_line_axis=request_line_axis,
            request_line_index=request_line_index,
            agent_starts=agent_starts,
        ),
    }


def _build_points(layout_name, system, n_agents):
    if layout_name == "baseline":
        return build_baseline_points(n_agents=n_agents)
    return build_paper_points(system=system.value, n_agents=n_agents)


def _make_controller(system, env, points, service_steps, stall_threshold):
    if system == SystemType.DSD:
        return DSDController(
            env,
            points,
            service_steps=service_steps,
            stall_threshold=stall_threshold,
        )
    return FSDTriggerController(
        env,
        points,
        service_steps=service_steps,
        stall_threshold=stall_threshold,
    )


def _sample_target(
    rng,
    points,
    workload,
    request_line_axis="aisle",
    request_line_index=None,
):
    candidates = _target_candidates(
        points,
        workload,
        request_line_axis=request_line_axis,
        request_line_index=request_line_index,
    )
    idx = int(rng.integers(0, len(candidates)))
    return candidates[idx]


def _target_candidates(
    points,
    workload,
    request_line_axis="aisle",
    request_line_index=None,
):
    if workload == "uniform":
        return points.storage_points

    if workload == "hotspot":
        middle = len(points.aisle_columns) // 2
        hot_aisles = set(points.aisle_columns[middle:middle + 2])
        candidates = [
            point
            for point in points.storage_points
            if point[0] in hot_aisles
        ]
        return candidates or points.storage_points

    if request_line_axis == "row":
        rows = sorted({point[1] for point in points.storage_points})
        row = _select_line_value(rows, request_line_index)
        candidates = [point for point in points.storage_points if point[1] == row]
    else:
        aisle = _select_line_value(points.aisle_columns, request_line_index)
        candidates = [point for point in points.storage_points if point[0] == aisle]
    return candidates or points.storage_points


def _select_line_value(values, requested):
    if requested is None:
        return values[len(values) // 2]
    if requested in values:
        return requested
    idx = max(0, min(len(values) - 1, int(requested)))
    return values[idx]


def _sample_kind(rng):
    if rng.random() < 0.5:
        return TransactionKind.STORAGE
    return TransactionKind.RETRIEVAL


def _stage_agents(env, points, n_agents, agent_starts=None):
    starts = _normalise_agent_starts(agent_starts, points, n_agents)
    for agent, (x, y, direction) in zip(env.unwrapped.agents, starts):
        agent.x = x
        agent.y = y
        agent.dir = getattr(Direction, direction)
        agent.carrying_shelf = None
        agent.has_delivered = False
    env.unwrapped._recalc_grid()
    env.unwrapped.request_queue = []
    env.unwrapped._agent_wait_steps[:] = 0
    env.unwrapped._position_history.clear()


def _normalise_agent_starts(agent_starts, points, n_agents):
    auto_starts = _start_positions(points, n_agents)
    if not agent_starts:
        return auto_starts

    starts = list(agent_starts[:n_agents])
    if len(starts) < n_agents:
        starts.extend(auto_starts[len(starts):])
    return starts


def _record_health_metrics(env, controller):
    wait_steps = list(getattr(env.unwrapped, "_agent_wait_steps", []))
    controller.stats["max_env_wait"] = max(
        controller.stats.get("max_env_wait", 0),
        max(wait_steps, default=0),
    )
    active_wait = [
        wait_steps[idx]
        for idx in controller.agent_tx
        if idx < len(wait_steps) and idx not in controller.service_until
    ]
    controller.stats["max_active_wait"] = max(
        controller.stats.get("max_active_wait", 0),
        max(active_wait, default=0),
    )


def _normalise_intentional_waits(env, controller):
    wait_steps = getattr(env.unwrapped, "_agent_wait_steps", None)
    if wait_steps is None:
        return

    for idx in range(len(wait_steps)):
        parked_without_task = idx not in controller.agent_tx and idx not in controller.recovery_targets
        servicing = idx in controller.service_until
        if parked_without_task or servicing:
            wait_steps[idx] = 0

    metrics = getattr(env.unwrapped, "_last_info", {}).get("metrics", {})
    if metrics:
        metrics["avg_wait_steps"] = float(np.mean(wait_steps)) if len(wait_steps) else 0.0
        metrics["stalled_agents"] = int(np.sum(wait_steps > 0))


def _start_positions(points, n_agents):
    starts = []
    used = set()
    candidate_rows = sorted(points.transition_rows)
    candidate_points = sorted(
        points.decision_points,
        key=lambda point: (
            candidate_rows.index(point[1]) if point[1] in candidate_rows else 99,
            point[0],
        ),
    )
    for idx in range(n_agents):
        if idx < len(points.zones) and points.zones[idx]:
            zone_xs = {cell[0] for cell in points.zones[idx]}
            candidates = [
                point
                for point in candidate_points
                if min(abs(point[0] - zone_x) for zone_x in zone_xs) <= 2
                and point not in used
            ]
        else:
            candidates = [point for point in candidate_points if point not in used]
        if not candidates:
            candidates = [point for point in candidate_points if point not in used]
        x, y = candidates[0]
        used.add((x, y))
        starts.append((x, y, "DOWN"))
    return starts


def parse_args():
    parser = argparse.ArgumentParser(description="Run DSD/FSD baseline on RWARE.")
    parser.add_argument("--system", choices=["dsd", "fsd", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--arrival-interval", type=int, default=5)
    parser.add_argument("--stall-threshold", type=int, default=8)
    parser.add_argument("--layout", choices=["paper", "baseline"], default="paper")
    parser.add_argument("--workload", choices=["uniform", "hotspot", "line"], default="uniform")
    parser.add_argument("--request-line-axis", choices=["aisle", "row"], default="aisle")
    parser.add_argument(
        "--request-line-index",
        type=int,
        default=None,
        help="Line index or actual x/y coordinate used when --workload line is selected.",
    )
    parser.add_argument(
        "--agent-starts",
        default=None,
        help="Optional starts as 'x,y,DIR;x,y,DIR'. Missing agents use automatic starts.",
    )
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    agent_starts = _parse_agent_starts(args.agent_starts)
    if args.system == "both":
        result = run_comparison(
            seed=args.seed,
            max_steps=args.steps,
            n_agents=args.agents,
            arrival_interval=args.arrival_interval,
            stall_threshold=args.stall_threshold,
            layout_name=args.layout,
            workload=args.workload,
            request_line_axis=args.request_line_axis,
            request_line_index=args.request_line_index,
            agent_starts=agent_starts,
        )
    else:
        result = run_once(
            args.system,
            seed=args.seed,
            max_steps=args.steps,
            n_agents=args.agents,
            arrival_interval=args.arrival_interval,
            stall_threshold=args.stall_threshold,
            layout_name=args.layout,
            workload=args.workload,
            request_line_axis=args.request_line_axis,
            request_line_index=args.request_line_index,
            agent_starts=agent_starts,
            render=args.render,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


def _parse_agent_starts(raw):
    if not raw:
        return None

    starts = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(",")]
        if len(parts) not in {2, 3}:
            raise ValueError("--agent-starts must use 'x,y,DIR;x,y,DIR'")
        direction = parts[2].upper() if len(parts) == 3 else "DOWN"
        if direction not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError(f"Unknown direction in --agent-starts: {direction}")
        starts.append((int(parts[0]), int(parts[1]), direction))
    return starts or None


if __name__ == "__main__":
    main()
