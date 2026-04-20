import argparse
import warnings

warnings.filterwarnings("ignore")

from deadlock.scenarios import dsd_fsd_baseline, factory_complex, factory_complex_gnn, factory_floor
from deadlock.scenarios.intersection import human_blockage, intersection
from deadlock.scenarios.single import single_lane


SCENARIOS = {
    "single_lane": single_lane.run,
    "intersection": intersection.run,
    "human_blockage": human_blockage.run,
    "factory_floor": factory_floor.run,
    "factory_complex": factory_complex.run,
    "factory_complex_gnn": factory_complex_gnn.run,
    "dsd_fsd_baseline": dsd_fsd_baseline.run,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Robotic warehouse deadlock scenarios with human-aware metrics"
    )
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="all",
        help="Scenario to execute",
    )
    parser.add_argument("--seed", type=int, default=0, help="Reset seed")
    parser.add_argument("--steps", type=int, default=80, help="Max rollout steps")
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce per-step console logs"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open the pyglet renderer with overlays enabled",
    )
    parser.add_argument(
        "--hold",
        action="store_true",
        help="Keep the render window open after the rollout ends",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save rollout artifacts",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="Base file name for saved rollout artifacts",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Output GIF frame rate",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    kwargs = {
        "seed": args.seed,
        "max_steps": args.steps,
        "verbose": not args.quiet,
        "render": args.render,
        "hold": args.hold,
        "save_dir": args.save_dir,
        "save_name": args.save_name,
        "fps": args.fps,
    }

    targets = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in targets:
        SCENARIOS[name](**kwargs)


if __name__ == "__main__":
    main()
