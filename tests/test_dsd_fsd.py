from deadlock.dsd_fsd.experiment import run_comparison, run_once
from deadlock.dsd_fsd.layouts import build_baseline_points, build_paper_points
from deadlock.dsd_fsd.model import SystemType
from deadlock.dsd_fsd.research_results import RESEARCH_SCENARIOS, run_research_scenario


def test_baseline_points_have_required_roles():
    points = build_baseline_points(n_agents=3)

    assert points.decision_points
    assert points.waiting_points
    assert points.escape_points
    assert points.conflict_zones
    assert len(points.zones) == 3


def test_paper_layout_uses_one_cell_aisle_gaps():
    points = build_paper_points(system="fsd", aisles=2)
    first_row = points.layout.splitlines()[0]

    assert first_row == "xx.xx.xx"
    assert points.aisle_columns == [2, 5]


def test_dsd_fsd_short_runs_produce_metrics():
    result = run_comparison(seed=1, max_steps=40, n_agents=3, arrival_interval=4)

    for system in ("dsd", "fsd"):
        metrics = result[system]
        assert metrics["created"] > 0
        assert metrics["completed"] >= 0
        assert "f_avg" in metrics
        assert "f_max" in metrics
        assert "wr" in metrics


def test_fsd_controller_records_trigger_fields():
    metrics = run_once(
        SystemType.FSD,
        seed=2,
        max_steps=40,
        n_agents=3,
        arrival_interval=3,
    )

    assert "triggers" in metrics
    assert "forced_escapes" in metrics
    assert "stall_recoveries" in metrics
    assert "local_escapes" in metrics


def test_stall_recovery_can_be_enabled_aggressively():
    metrics = run_once(
        SystemType.FSD,
        seed=0,
        max_steps=80,
        n_agents=3,
        arrival_interval=3,
        stall_threshold=2,
    )

    assert metrics["stall_recoveries"] > 0


def test_fsd_1000_step_run_keeps_active_stalls_bounded():
    metrics = run_once(
        SystemType.FSD,
        seed=0,
        max_steps=1000,
        n_agents=3,
        arrival_interval=5,
        stall_threshold=6,
        layout_name="baseline",
    )

    assert metrics["completed"] >= 100
    assert metrics["max_active_wait"] <= 10
    assert metrics["recovery_expired"] == 0


def test_hotspot_workload_exposes_fsd_bottleneck_relief():
    result = run_comparison(
        seed=0,
        max_steps=1000,
        n_agents=4,
        arrival_interval=5,
        stall_threshold=8,
        layout_name="paper",
        workload="hotspot",
    )

    assert result["fsd"]["completed"] > result["dsd"]["completed"]
    assert result["fsd"]["f_avg"] < result["dsd"]["f_avg"]


def test_local_graph_admission_reduces_shared_queue_blocking():
    rule_metrics = run_research_scenario(RESEARCH_SCENARIOS[2], steps=120)
    gnn_metrics = run_research_scenario(RESEARCH_SCENARIOS[3], steps=120)

    assert gnn_metrics["max_active_wait"] < rule_metrics["max_active_wait"]
    assert gnn_metrics["graph_decisions"] > 0
