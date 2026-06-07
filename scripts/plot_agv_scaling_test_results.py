from __future__ import annotations

import argparse
import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_DIR / "final_output" / "test_100_by_agv_fixed_weights"
DEFAULT_DESKTOP_DIR = Path.home() / "OneDrive" / "바탕 화면" / "agv_scaling_plots"

POLICY_LABELS = {
    "fcfs": "FCFS",
    "fixed_heuristic": "Basic-Heuristic",
    "bo_heuristic": "BO-Heuristic",
    "pso_heuristic": "PSO-Heuristic",
}
POLICY_ORDER = ["fcfs", "fixed_heuristic", "bo_heuristic", "pso_heuristic"]
POLICY_COLORS = {
    "fcfs": "#6b7280",
    "fixed_heuristic": "#d97706",
    "bo_heuristic": "#2563eb",
    "pso_heuristic": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AGV scaling analysis for fixed-weight test results."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--copy-to-desktop",
        action="store_true",
        help="Also copy generated PNGs/CSVs to a desktop folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    plot_dir = result_dir / "agv_scaling_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    by_agv = _read_rows(result_dir / "test_policy_summary_by_agv.csv")
    by_case_agv = _read_rows(result_dir / "test_scenario_case_summary_by_agv.csv")
    raw_runs = _read_rows(result_dir / "test_raw_runs.csv")

    metrics_rows = _build_scaling_metrics(by_agv)
    _write_csv(metrics_rows, plot_dir / "agv_scaling_metrics.csv")

    _plot_avg_total_time(by_agv, plot_dir)
    _plot_improvement_vs_fcfs(by_agv, plot_dir)
    _plot_delta_16_to_24(metrics_rows, plot_dir)
    _plot_distribution(raw_runs, plot_dir)
    _plot_case_scaling(by_case_agv, plot_dir)

    _write_analysis_markdown(metrics_rows, by_agv, by_case_agv, plot_dir)

    if args.copy_to_desktop:
        DEFAULT_DESKTOP_DIR.mkdir(parents=True, exist_ok=True)
        for path in plot_dir.iterdir():
            if path.is_file():
                shutil.copy2(path, DEFAULT_DESKTOP_DIR / path.name)

    print(f"wrote plots: {plot_dir.resolve()}")
    if args.copy_to_desktop:
        print(f"copied plots to desktop: {DEFAULT_DESKTOP_DIR}")


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    value = row[key]
    if value == "":
        return math.nan
    return float(value)


def _build_scaling_metrics(rows: list[dict]) -> list[dict]:
    by_policy_agv = {
        (row["policy"], int(row["total_agvs"])): _float(row, "avg_total_time")
        for row in rows
    }
    fcfs_by_agv = {
        agv: by_policy_agv[("fcfs", agv)]
        for agv in (16, 20, 24)
    }

    metrics = []
    for policy in POLICY_ORDER:
        avg16 = by_policy_agv[(policy, 16)]
        avg20 = by_policy_agv[(policy, 20)]
        avg24 = by_policy_agv[(policy, 24)]
        row = {
            "policy": policy,
            "policy_label": POLICY_LABELS[policy],
            "avg_16": avg16,
            "avg_20": avg20,
            "avg_24": avg24,
            "delta_16_to_20": avg20 - avg16,
            "delta_20_to_24": avg24 - avg20,
            "delta_16_to_24": avg24 - avg16,
            "pct_increase_16_to_24": (avg24 - avg16) / avg16 * 100.0,
            "improvement_vs_fcfs_16_pct": (fcfs_by_agv[16] - avg16) / fcfs_by_agv[16] * 100.0,
            "improvement_vs_fcfs_20_pct": (fcfs_by_agv[20] - avg20) / fcfs_by_agv[20] * 100.0,
            "improvement_vs_fcfs_24_pct": (fcfs_by_agv[24] - avg24) / fcfs_by_agv[24] * 100.0,
        }
        metrics.append(row)
    return metrics


def _plot_avg_total_time(rows: list[dict], plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for policy in POLICY_ORDER:
        policy_rows = sorted(
            [row for row in rows if row["policy"] == policy],
            key=lambda row: int(row["total_agvs"]),
        )
        xs = [int(row["total_agvs"]) for row in policy_rows]
        ys = [_float(row, "avg_total_time") for row in policy_rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.4,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)

    ax.set_title("Average Total_Time as AGV Count Increases")
    ax.set_xlabel("AGV count")
    ax.set_ylabel("Average Total_Time")
    ax.set_xticks([16, 20, 24])
    ax.grid(True, alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_avg_total_time_by_agv_policy.png", dpi=180)
    plt.close(fig)


def _plot_improvement_vs_fcfs(rows: list[dict], plot_dir: Path) -> None:
    fcfs = {
        int(row["total_agvs"]): _float(row, "avg_total_time")
        for row in rows
        if row["policy"] == "fcfs"
    }
    agvs = [16, 20, 24]
    heuristic_policies = ["fixed_heuristic", "bo_heuristic", "pso_heuristic"]
    width = 0.22
    offsets = [-width, 0.0, width]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for index, policy in enumerate(heuristic_policies):
        values = []
        for agv in agvs:
            policy_avg = next(
                _float(row, "avg_total_time")
                for row in rows
                if row["policy"] == policy and int(row["total_agvs"]) == agv
            )
            values.append((fcfs[agv] - policy_avg) / fcfs[agv] * 100.0)
        x_positions = [agv + offsets[index] for agv in agvs]
        bars = ax.bar(
            x_positions,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
        )
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=9)

    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("Improvement over FCFS by AGV Count")
    ax.set_xlabel("AGV count")
    ax.set_ylabel("Improvement vs FCFS (%)")
    ax.set_xticks(agvs)
    ax.set_ylim(0, max(ax.get_ylim()[1], 14))
    ax.grid(True, axis="y", alpha=0.28)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "02_improvement_vs_fcfs_by_agv.png", dpi=180)
    plt.close(fig)


def _plot_delta_16_to_24(metrics_rows: list[dict], plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    labels = [row["policy_label"] for row in metrics_rows]
    values = [float(row["delta_16_to_24"]) for row in metrics_rows]
    colors = [POLICY_COLORS[row["policy"]] for row in metrics_rows]
    bars = ax.bar(labels, values, color=colors)
    ax.bar_label(bars, labels=[f"+{value:.2f}" for value in values], padding=3, fontsize=9)
    ax.set_title("Total_Time Increase from 16 AGVs to 24 AGVs")
    ax.set_ylabel("Average Total_Time increase")
    ax.grid(True, axis="y", alpha=0.28)
    fig.tight_layout()
    fig.savefig(plot_dir / "03_delta_16_to_24_by_policy.png", dpi=180)
    plt.close(fig)


def _plot_distribution(rows: list[dict], plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), sharey=True)
    for ax, agv in zip(axes, [16, 20, 24]):
        data = []
        labels = []
        for policy in POLICY_ORDER:
            values = [
                _float(row, "total_time")
                for row in rows
                if row["policy"] == policy and int(row["total_agvs"]) == agv
            ]
            data.append(values)
            labels.append(POLICY_LABELS[policy].replace("-Heuristic", ""))
        box = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
        for patch, policy in zip(box["boxes"], POLICY_ORDER):
            patch.set_facecolor(POLICY_COLORS[policy])
            patch.set_alpha(0.45)
        ax.set_title(f"AGV {agv}")
        ax.set_xlabel("Policy")
        ax.grid(True, axis="y", alpha=0.28)
        ax.tick_params(axis="x", rotation=18)
    axes[0].set_ylabel("Total_Time")
    fig.suptitle("Total_Time Distribution by AGV Count and Policy", y=1.03)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_total_time_distribution_boxplot.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_case_scaling(rows: list[dict], plot_dir: Path) -> None:
    case_ids = sorted({int(row["scenario_case_id"]) for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), sharex=True)
    axes_flat = axes.flatten()

    for ax, case_id in zip(axes_flat, case_ids):
        case_name = next(
            row["scenario_case_name"]
            for row in rows
            if int(row["scenario_case_id"]) == case_id
        )
        for policy in POLICY_ORDER:
            policy_rows = sorted(
                [
                    row
                    for row in rows
                    if int(row["scenario_case_id"]) == case_id and row["policy"] == policy
                ],
                key=lambda row: int(row["total_agvs"]),
            )
            xs = [int(row["total_agvs"]) for row in policy_rows]
            ys = [_float(row, "avg_total_time") for row in policy_rows]
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2.0,
                color=POLICY_COLORS[policy],
                label=POLICY_LABELS[policy],
            )
        ax.set_title(f"Scenario {case_id}: {case_name}")
        ax.set_xticks([16, 20, 24])
        ax.grid(True, alpha=0.28)

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.96, 0.08))
    fig.supxlabel("AGV count")
    fig.supylabel("Average Total_Time")
    fig.suptitle("Scenario-wise Total_Time Scaling", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "05_scenario_case_scaling_by_agv.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_analysis_markdown(
    metrics_rows: list[dict],
    by_agv: list[dict],
    by_case_agv: list[dict],
    plot_dir: Path,
) -> None:
    best_scaling = min(metrics_rows, key=lambda row: float(row["delta_16_to_24"]))
    lines = [
        "# AGV Scaling Analysis",
        "",
        "## Key Scaling Metrics",
        "",
        "| policy | avg16 | avg20 | avg24 | delta16-24 | pct increase |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['policy_label']} | {row['avg_16']:.2f} | {row['avg_20']:.2f} | "
            f"{row['avg_24']:.2f} | +{row['delta_16_to_24']:.2f} | "
            f"{row['pct_increase_16_to_24']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"Lowest 16-to-24 increase: {best_scaling['policy_label']} "
            f"(+{best_scaling['delta_16_to_24']:.2f}).",
            "",
            "## Generated Figures",
            "",
            "- 01_avg_total_time_by_agv_policy.png",
            "- 02_improvement_vs_fcfs_by_agv.png",
            "- 03_delta_16_to_24_by_policy.png",
            "- 04_total_time_distribution_boxplot.png",
            "- 05_scenario_case_scaling_by_agv.png",
        ]
    )
    (plot_dir / "agv_scaling_analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.2f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


if __name__ == "__main__":
    main()
