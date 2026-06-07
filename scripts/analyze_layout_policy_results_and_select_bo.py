from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SINGLE_DIR = PROJECT_DIR / "final_output" / "test_100_by_agv_fixed_weights"
DEFAULT_DOUBLE_DIR = PROJECT_DIR / "final_output" / "double_cross_test_100_by_agv_fixed_weights"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "final_output" / "layout_policy_bo_advantage_analysis"
DEFAULT_DESKTOP_DIR = Path.home() / "OneDrive" / "바탕 화면" / "layout_policy_bo_advantage_analysis"

POLICY_COLUMNS = {
    "fcfs": "fcfs_total_time",
    "fixed_heuristic": "fixed_heuristic_total_time",
    "bo_heuristic": "bo_heuristic_total_time",
    "pso_heuristic": "pso_heuristic_total_time",
}
POLICY_LABELS = {
    "fcfs": "FCFS",
    "fixed_heuristic": "Basic",
    "bo_heuristic": "BO",
    "pso_heuristic": "PSO",
}
POLICY_COLORS = {
    "fcfs": "#6b7280",
    "fixed_heuristic": "#d97706",
    "bo_heuristic": "#2563eb",
    "pso_heuristic": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scenario-level policy result CSVs, graphs, and BO-advantage selections."
    )
    parser.add_argument("--single-dir", type=Path, default=DEFAULT_SINGLE_DIR)
    parser.add_argument("--double-dir", type=Path, default=DEFAULT_DOUBLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--copy-to-desktop", action="store_true")
    parser.add_argument("--desktop-dir", type=Path, default=DEFAULT_DESKTOP_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    single = _load_layout_bundle(
        layout_key="single_cross",
        result_dir=args.single_dir,
        comparison_file="test_trial_comparison.csv",
        raw_file="test_raw_runs.csv",
        scenario_file="test_scenarios_300_all_agv_counts.csv",
    )
    double = _load_layout_bundle(
        layout_key="double_cross",
        result_dir=args.double_dir,
        comparison_file="double_cross_test_trial_comparison.csv",
        raw_file="double_cross_test_raw_runs.csv",
        scenario_file="double_cross_test_scenarios_200_all_agv_counts.csv",
    )

    selected = [_select_best_bo_case(single["wide"]), _select_best_bo_case(double["wide"])]

    _write_csv(single["wide"], args.output_dir / "single_cross_scenario_policy_results_wide.csv")
    _write_csv(single["long"], args.output_dir / "single_cross_scenario_policy_results_long.csv")
    _write_csv(double["wide"], args.output_dir / "double_cross_scenario_policy_results_wide.csv")
    _write_csv(double["long"], args.output_dir / "double_cross_scenario_policy_results_long.csv")
    _write_csv(selected, args.output_dir / "selected_bo_advantage_cases.csv")

    _plot_policy_avg_by_case(single["long"], args.output_dir / "01_single_policy_avg_by_scenario_case.png", "Single Cross")
    _plot_policy_avg_by_case(double["long"], args.output_dir / "02_double_policy_avg_by_scenario_case.png", "Double Cross")
    _plot_top_bo_margins(single["wide"], args.output_dir / "03_single_top_bo_advantage_scenarios.png", "Single Cross")
    _plot_top_bo_margins(double["wide"], args.output_dir / "04_double_top_bo_advantage_scenarios.png", "Double Cross")
    _plot_layout_policy_overall([*single["long"], *double["long"]], args.output_dir / "05_layout_policy_overall_avg.png")

    _write_summary(selected, args.output_dir / "bo_advantage_selection_summary.md")

    if args.copy_to_desktop:
        args.desktop_dir.mkdir(parents=True, exist_ok=True)
        for path in args.output_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, args.desktop_dir / path.name)

    print(f"wrote analysis dir: {args.output_dir.resolve()}")
    for row in selected:
        print(
            f"{row['layout_key']} selected run={row['run_index']} "
            f"case={row['scenario_case_id']} AGV={row['total_agvs']} "
            f"BO margin={row['bo_margin_vs_best_other']}"
        )
    if args.copy_to_desktop:
        print(f"copied to desktop: {args.desktop_dir}")


def _load_layout_bundle(
    layout_key: str,
    result_dir: Path,
    comparison_file: str,
    raw_file: str,
    scenario_file: str,
) -> dict[str, list[dict]]:
    scenario_rows = {
        int(row["run_index"]): row
        for row in _read_csv(result_dir / scenario_file)
    }
    wide_rows = []
    for row in _read_csv(result_dir / comparison_file):
        scenario = scenario_rows[int(row["run_index"])]
        enriched = {
            "layout_key": layout_key,
            "source_result_dir": str(result_dir),
            "scenario_file": scenario_file,
            **row,
            "scenario_case_label": scenario.get("scenario_case_label", ""),
            "scenario_split": scenario.get("scenario_split", ""),
            "dominant_approach": scenario.get("dominant_approach", ""),
            "dominant_approach_count": scenario.get("dominant_approach_count", ""),
            "dominant_exit": scenario.get("dominant_exit", ""),
            "dominant_exit_count": scenario.get("dominant_exit_count", ""),
            "early_departures": scenario.get("early_departures", ""),
            "spawn_plan_json": scenario["spawn_plan_json"],
            "goal_plan_json": scenario["goal_plan_json"],
        }
        if "total_agvs" not in enriched or enriched["total_agvs"] == "":
            enriched["total_agvs"] = scenario["total_agvs"]
        _add_bo_margin(enriched)
        wide_rows.append(enriched)

    long_rows = []
    for row in _read_csv(result_dir / raw_file):
        scenario = scenario_rows[int(row["run_index"])]
        long_rows.append(
            {
                "layout_key": layout_key,
                "run_index": row["run_index"],
                "scenario_seed": row["scenario_seed"],
                "total_agvs": row["total_agvs"],
                "scenario_case_id": row["scenario_case_id"],
                "scenario_case_name": row["scenario_case_name"],
                "scenario_case_label": scenario.get("scenario_case_label", ""),
                "policy": row["policy"],
                "total_time": row["total_time"],
                "completed": row["completed"],
                "spawn_plan_json": row["spawn_plan_json"],
                "goal_plan_json": row["goal_plan_json"],
            }
        )
    return {"wide": wide_rows, "long": long_rows}


def _add_bo_margin(row: dict) -> None:
    bo = int(float(row["bo_heuristic_total_time"]))
    other_times = {
        policy: int(float(row[column]))
        for policy, column in POLICY_COLUMNS.items()
        if policy != "bo_heuristic"
    }
    best_other_policy, best_other_time = min(other_times.items(), key=lambda item: item[1])
    worst_other_policy, worst_other_time = max(other_times.items(), key=lambda item: item[1])
    row["best_other_policy"] = best_other_policy
    row["best_other_total_time"] = best_other_time
    row["worst_other_policy"] = worst_other_policy
    row["worst_other_total_time"] = worst_other_time
    row["bo_margin_vs_best_other"] = best_other_time - bo
    row["bo_margin_vs_fcfs"] = int(float(row["fcfs_total_time"])) - bo
    row["bo_margin_vs_basic"] = int(float(row["fixed_heuristic_total_time"])) - bo
    row["bo_margin_vs_pso"] = int(float(row["pso_heuristic_total_time"])) - bo
    row["bo_unique_best"] = int(row["bo_margin_vs_best_other"] > 0)


def _select_best_bo_case(rows: list[dict]) -> dict:
    positive = [row for row in rows if int(row["bo_margin_vs_best_other"]) > 0]
    candidates = positive or rows
    selected = max(
        candidates,
        key=lambda row: (
            int(row["bo_margin_vs_best_other"]),
            int(row["bo_margin_vs_fcfs"]),
            int(row["total_agvs"]),
        ),
    )
    return dict(selected)


def _plot_policy_avg_by_case(rows: list[dict], path: Path, title_prefix: str) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["scenario_case_id"]), row["scenario_case_name"], row["policy"])].append(float(row["total_time"]))

    case_ids = sorted({key[0] for key in grouped})
    policies = list(POLICY_COLUMNS)
    width = 0.19
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    for index, policy in enumerate(policies):
        values = []
        for case_id in case_ids:
            matching = [
                values
                for (current_case_id, _, current_policy), values in grouped.items()
                if current_case_id == case_id and current_policy == policy
            ]
            values.append(sum(matching[0]) / len(matching[0]))
        positions = [case_id + offsets[index] for case_id in case_ids]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)

    ax.set_title(f"{title_prefix}: Average Total_Time by Scenario Case")
    ax.set_xlabel("Scenario case")
    ax.set_ylabel("Average Total_Time")
    ax.set_xticks(case_ids)
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_top_bo_margins(rows: list[dict], path: Path, title_prefix: str) -> None:
    top_rows = sorted(rows, key=lambda row: int(row["bo_margin_vs_best_other"]), reverse=True)[:20]
    labels = [
        f"r{row['run_index']}\nS{row['scenario_case_id']},A{row['total_agvs']}"
        for row in top_rows
    ]
    values = [int(row["bo_margin_vs_best_other"]) for row in top_rows]

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    bars = ax.bar(labels, values, color="#2563eb")
    ax.bar_label(bars, labels=[str(value) for value in values], padding=2, fontsize=8)
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title(f"{title_prefix}: Top BO Advantage Scenarios")
    ax.set_xlabel("Run / Scenario / AGV")
    ax.set_ylabel("BO margin vs best other policy")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_layout_policy_overall(rows: list[dict], path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["layout_key"], row["policy"])].append(float(row["total_time"]))

    layout_keys = ["single_cross", "double_cross"]
    policies = list(POLICY_COLUMNS)
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for index, policy in enumerate(policies):
        values = [
            sum(grouped[(layout_key, policy)]) / len(grouped[(layout_key, policy)])
            for layout_key in layout_keys
        ]
        positions = [position + offsets[index] for position in range(len(layout_keys))]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=2, fontsize=8)
    ax.set_title("Overall Average Total_Time by Layout and Policy")
    ax.set_xticks(range(len(layout_keys)))
    ax.set_xticklabels(["Single Cross", "Double Cross"])
    ax.set_ylabel("Average Total_Time")
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(selected: list[dict], path: Path) -> None:
    lines = [
        "# BO Advantage Selection",
        "",
        "| layout | run | AGV | case | BO | best other | margin |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            f"| {row['layout_key']} | {row['run_index']} | {row['total_agvs']} | "
            f"{row['scenario_case_id']} | {row['bo_heuristic_total_time']} | "
            f"{row['best_other_total_time']} ({row['best_other_policy']}) | "
            f"{row['bo_margin_vs_best_other']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    discovered = {key for row in rows for key in row}
    fieldnames.extend(key for key in sorted(discovered) if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
