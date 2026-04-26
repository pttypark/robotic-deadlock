from dataclasses import replace

from deadlock.dsd_fsd.benchmark.config import ExperimentConfig
from deadlock.dsd_fsd.benchmark.experiment_runner import run_experiment_grid
from deadlock.dsd_fsd.benchmark.layout_builder import build_layout
from deadlock.dsd_fsd.benchmark.simulator import WarehouseSimulator, generate_task_arrivals


def test_benchmark_layout_is_deterministic_and_shared():
    config = ExperimentConfig(policy_type="rule_based", num_robots=4, num_aisles=6, seed=2)
    layout_a = build_layout(config)
    layout_b = build_layout(replace(config, policy_type="fsd"))

    assert layout_a.layout == layout_b.layout
    assert layout_a.aisle_columns == layout_b.aisle_columns
    assert layout_a.transition_rows == layout_b.transition_rows
    assert layout_a.decision_points
    assert layout_a.waiting_points
    assert layout_a.escape_points


def test_task_sequence_ignores_policy_type_for_fair_comparison():
    base = ExperimentConfig(policy_type="rule_based", num_robots=3, num_aisles=6, seed=4)
    layout = build_layout(base)

    rule_arrivals = generate_task_arrivals(base, layout)
    fsd_arrivals = generate_task_arrivals(replace(base, policy_type="fsd"), layout)

    assert rule_arrivals == fsd_arrivals


def test_dsd_allowed_cells_do_not_cross_zone_boundary():
    config = ExperimentConfig(policy_type="dsd", num_robots=3, num_aisles=6)
    layout = build_layout(config)

    for zone_id, allowed in enumerate(layout.zone_allowed):
        zone = layout.zones[zone_id]
        if not zone:
            continue
        zone_xs = {cell[0] for cell in zone}
        lo = min(zone_xs) - 1
        hi = max(zone_xs) + 1
        assert all(lo <= cell[0] <= hi for cell in allowed)


def test_benchmark_simulator_returns_required_kpis():
    config = ExperimentConfig(
        policy_type="fsd",
        num_robots=4,
        num_aisles=6,
        capacity="small",
        arrival_rate="high",
        seed=0,
        max_episode_steps=120,
    )
    simulator = WarehouseSimulator(config)
    metrics = simulator.run()

    for column in (
        "Favg",
        "Fmax",
        "Throughput",
        "BlockingTime",
        "AvgQueueLength",
        "DeadlockCount",
        "TriggerCount",
        "RerouteCount",
    ):
        assert column in metrics
    assert metrics["Throughput"] >= 0


def test_experiment_runner_writes_csv_and_plots(tmp_path):
    rows = run_experiment_grid(
        seeds=[0],
        policies=["rule_based", "dsd", "fsd"],
        robot_counts=[3],
        aisle_counts=[6],
        capacities=["small"],
        arrival_rates=["medium"],
        max_episode_steps=60,
        output_dir=tmp_path,
    )

    assert len(rows) == 3
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary_table.csv").exists()
    assert (tmp_path / "plots" / "Favg_comparison.svg").exists()
