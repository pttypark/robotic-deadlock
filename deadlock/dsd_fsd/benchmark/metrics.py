from __future__ import annotations

import csv
import statistics
from pathlib import Path


REQUIRED_RESULT_COLUMNS = [
    "seed",
    "policy_type",
    "num_robots",
    "num_aisles",
    "capacity",
    "arrival_rate",
    "Favg",
    "Fmax",
    "Throughput",
    "BlockingTime",
    "AvgQueueLength",
    "DeadlockCount",
    "TriggerCount",
    "RerouteCount",
]

EXTRA_RESULT_COLUMNS = [
    "CollisionAttempts",
    "ForcedRelocationCount",
    "RobotUtilization",
    "AverageTravelDistance",
]


def write_results_csv(rows: list[dict], path: Path) -> None:
    """Write per-episode KPI rows.

    Args:
        rows: Flat KPI dictionaries.
        path: Destination CSV path.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = REQUIRED_RESULT_COLUMNS + [
        column
        for column in EXTRA_RESULT_COLUMNS
        if any(column in row for row in rows)
    ]
    dynamic_columns = sorted(
        {
            column
            for row in rows
            for column in row
            if column not in columns
        }
    )
    columns.extend(dynamic_columns)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_summary_table(rows: list[dict], path: Path) -> list[dict]:
    """Write policy-level KPI mean/std summary.

    Args:
        rows: Per-episode KPI rows.
        path: Destination CSV path.

    Returns:
        Summary rows written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        "Favg",
        "Fmax",
        "Throughput",
        "BlockingTime",
        "AvgQueueLength",
        "DeadlockCount",
        "TriggerCount",
        "RerouteCount",
    ]
    policies = sorted({row["policy_type"] for row in rows})
    summary_rows = []
    for policy in policies:
        policy_rows = [row for row in rows if row["policy_type"] == policy]
        summary = {"policy_type": policy, "n": len(policy_rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in policy_rows]
            summary[f"{metric}_mean"] = statistics.mean(values) if values else 0.0
            summary[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    columns = ["policy_type", "n"]
    for metric in metrics:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)
    return summary_rows


def write_plots(rows: list[dict], output_dir: Path) -> list[Path]:
    """Write simple dependency-free SVG policy comparison plots.

    Args:
        rows: Per-episode KPI rows.
        output_dir: Directory where plot SVGs are written.

    Returns:
        List of plot paths created.
    """

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("Favg", "Favg comparison"),
        ("Fmax", "Fmax comparison"),
        ("Throughput", "Throughput comparison"),
        ("BlockingTime", "Blocking time comparison"),
        ("AvgQueueLength", "Queue length comparison"),
    ]
    policies = sorted({row["policy_type"] for row in rows})
    created = []
    for metric, title in metrics:
        values = []
        for policy in policies:
            policy_values = [float(row[metric]) for row in rows if row["policy_type"] == policy]
            values.append(statistics.mean(policy_values) if policy_values else 0.0)
        path = plot_dir / f"{metric}_comparison.svg"
        path.write_text(_svg_bar_chart(title, metric, policies, values), encoding="utf-8")
        created.append(path)
    return created


def _svg_bar_chart(title: str, metric: str, labels: list[str], values: list[float]) -> str:
    width = 640
    height = 420
    margin_left = 72
    margin_right = 28
    margin_top = 54
    margin_bottom = 72
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    max_value = max(values) if values else 0.0
    scale_max = max(1.0, max_value * 1.15)
    bar_gap = 26
    bar_width = max(36, (chart_width - bar_gap * (len(labels) + 1)) / max(1, len(labels)))
    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#72b7b2"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="18" font-weight="700">{_escape(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#222" stroke-width="1"/>',
        f'<text x="18" y="{margin_top + chart_height / 2}" transform="rotate(-90 18 {margin_top + chart_height / 2})" text-anchor="middle" font-family="Arial" font-size="13">{_escape(metric)}</text>',
    ]
    for tick_idx in range(5):
        ratio = tick_idx / 4
        value = scale_max * ratio
        y = margin_top + chart_height - chart_height * ratio
        elements.append(f'<line x1="{margin_left - 4}" y1="{y:.1f}" x2="{margin_left}" y2="{y:.1f}" stroke="#222"/>')
        elements.append(
            f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.1f}</text>'
        )
        if tick_idx:
            elements.append(
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>'
            )
    for idx, (label, value) in enumerate(zip(labels, values)):
        x = margin_left + bar_gap + idx * (bar_width + bar_gap)
        bar_height = chart_height * (value / scale_max)
        y = margin_top + chart_height - bar_height
        elements.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{colors[idx % len(colors)]}"/>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>'
        )
        elements.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{margin_top + chart_height + 24}" text-anchor="middle" font-family="Arial" font-size="12">{_escape(label)}</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements)


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
