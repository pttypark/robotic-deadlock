from __future__ import annotations

import argparse
from dataclasses import replace

from deadlock.dsd_fsd.benchmark.config import ExperimentConfig
from deadlock.dsd_fsd.benchmark.layout_builder import build_layout


def render_layout(config: ExperimentConfig, overlay: str = "roles") -> str:
    """Render benchmark layout as deterministic ASCII for inspection.

    Args:
        config: Experiment condition used to build the layout.
        overlay: One of base, roles, or zones.

    Returns:
        ASCII layout string.
    """

    layout = build_layout(config)
    rows = [list(row) for row in layout.layout.splitlines()]
    if overlay == "roles":
        for x, y in layout.escape_points:
            rows[y][x] = "E"
        for x, y in layout.waiting_points:
            rows[y][x] = "W"
        for x, y in layout.decision_points:
            rows[y][x] = "D"
        for x, y in layout.workstations:
            rows[y][x] = "G"
    elif overlay == "zones":
        for zone_id, zone in enumerate(layout.zones):
            marker = str(zone_id % 10)
            for x, y in zone:
                rows[y][x] = marker
        for x, y in layout.workstations:
            rows[y][x] = "G"
    elif overlay != "base":
        raise ValueError(f"Unknown overlay: {overlay}")

    legend = [
        f"policy={config.policy_type} robots={config.num_robots} aisles={config.num_aisles} capacity={config.capacity}",
        "legend: x=shelf/storage, .=road, G=workstation, D=decision, W=waiting, E=escape, 0-9=DSD zone",
    ]
    return "\n".join(legend + [""] + ["".join(row) for row in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print deterministic benchmark layout.")
    parser.add_argument("--policy", choices=["rule_based", "dsd", "fsd"], default="fsd")
    parser.add_argument("--robots", type=int, default=4)
    parser.add_argument("--aisles", type=int, default=6)
    parser.add_argument("--capacity", choices=["small", "large"], default="small")
    parser.add_argument("--arrival-rate", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overlay", choices=["base", "roles", "zones"], default="roles")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        policy_type=args.policy,
        num_robots=args.robots,
        num_aisles=args.aisles,
        capacity=args.capacity,
        arrival_rate=args.arrival_rate,
        seed=args.seed,
    )
    if args.overlay == "zones":
        config = replace(config, policy_type="dsd")
    print(render_layout(config, overlay=args.overlay))


if __name__ == "__main__":
    main()
