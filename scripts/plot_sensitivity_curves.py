"""Plot sensitivity curves by parameter."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_INPUT = (
    Path("final_output")
    / "basic_one_factor_sensitivity"
    / "sensitivity_feature_weight_statistics.csv"
)
DEFAULT_OUTPUT_DIR = Path("outputs") / "sensitivity_plots"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Total_Time sensitivity plots for each parameter."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    rows_by_feature = _load_rows(Path(args.input))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        Image = None
        ImageDraw = None
        ImageFont = None

    saved = []
    for feature, rows in sorted(rows_by_feature.items()):
        rows = sorted(rows, key=lambda row: row["weight"])
        if plt is None:
            if Image is None:
                output_path = output_dir / f"sensitivity_curve_{feature}.svg"
                _write_svg_curve(output_path, feature, rows)
            else:
                output_path = output_dir / f"sensitivity_curve_{feature}.png"
                _write_pil_curve(output_path, feature, rows, Image, ImageDraw, ImageFont)
        else:
            output_path = output_dir / f"sensitivity_curve_{feature}.png"
            _write_matplotlib_curve(output_path, feature, rows, plt)
        saved.append(output_path)

    print(f"saved {len(saved)} plots to {output_dir.resolve()}")
    for path in saved:
        print(path.resolve())


def _load_rows(path: Path) -> dict[str, list[dict]]:
    rows_by_feature: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows_by_feature[row["feature"]].append(
                {
                    "feature": row["feature"],
                    "weight": float(row["weight"]),
                    "mean_total_time": float(row["mean_total_time"]),
                    "std_total_time": float(row["std_total_time"]),
                }
            )
    return rows_by_feature


def _write_matplotlib_curve(output_path: Path, feature: str, rows: list[dict], plt) -> None:
    weights = [row["weight"] for row in rows]
    means = [row["mean_total_time"] for row in rows]
    stds = [row["std_total_time"] for row in rows]
    lower = [mean - std for mean, std in zip(means, stds)]
    upper = [mean + std for mean, std in zip(means, stds)]
    marker = _transition_best_point(rows)
    best_weight = marker["weight"]
    best_mean = marker["mean_total_time"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)
    ax.plot(weights, means, color="#1f77b4", linewidth=2.0, label="Mean Total_Time")
    ax.fill_between(weights, lower, upper, color="#1f77b4", alpha=0.15, label="Mean +/- 1 Std")
    ax.scatter([best_weight], [best_mean], color="#d62728", s=50, zorder=3, label=f"Transition best: {best_weight:.2f}, {best_mean:.2f}")
    ax.axvline(best_weight, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(f"Sensitivity Curve: {feature}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Total_Time")
    ax.set_xlim(-10, 10)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _write_svg_curve(output_path: Path, feature: str, rows: list[dict]) -> None:
    width = 1000
    height = 620
    left = 88
    right = 32
    top = 64
    bottom = 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    weights = [row["weight"] for row in rows]
    means = [row["mean_total_time"] for row in rows]
    stds = [row["std_total_time"] for row in rows]
    lower_values = [mean - std for mean, std in zip(means, stds)]
    upper_values = [mean + std for mean, std in zip(means, stds)]
    y_min = math.floor(min(lower_values) - 1.0)
    y_max = math.ceil(max(upper_values) + 1.0)
    marker = _transition_best_point(rows)
    best_weight = marker["weight"]
    best_mean = marker["mean_total_time"]

    def x_pos(weight: float) -> float:
        return left + (weight + 10.0) / 20.0 * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    mean_points = " ".join(
        f"{x_pos(weight):.2f},{y_pos(mean):.2f}"
        for weight, mean in zip(weights, means)
    )
    band_points = " ".join(
        f"{x_pos(weight):.2f},{y_pos(value):.2f}"
        for weight, value in zip(weights, upper_values)
    )
    band_points += " "
    band_points += " ".join(
        f"{x_pos(weight):.2f},{y_pos(value):.2f}"
        for weight, value in reversed(list(zip(weights, lower_values)))
    )

    x_ticks = [-10, -5, 0, 5, 10]
    y_step = max(1, math.ceil((y_max - y_min) / 8))
    y_start = math.ceil(y_min / y_step) * y_step
    y_ticks = list(range(y_start, y_max + 1, y_step))
    safe_title = html.escape(f"Sensitivity Curve: {feature}")
    safe_feature = html.escape(feature)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{safe_title}</text>',
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial" font-size="15">Weight</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" transform="rotate(-90 22 {height / 2})" font-family="Arial" font-size="15">Total_Time</text>',
    ]
    for tick in x_ticks:
        x = x_pos(tick)
        lines.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" stroke="#e6e6e6" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle" font-family="Arial" font-size="12">{tick}</text>',
            ]
        )
    for tick in y_ticks:
        y = y_pos(tick)
        lines.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>',
            ]
        )
    lines.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#333" stroke-width="1.2"/>',
            f'<polygon points="{band_points}" fill="#1f77b4" opacity="0.15"/>',
            f'<polyline points="{mean_points}" fill="none" stroke="#1f77b4" stroke-width="2.4"/>',
            f'<line x1="{x_pos(best_weight):.2f}" y1="{top}" x2="{x_pos(best_weight):.2f}" y2="{height - bottom}" stroke="#d62728" stroke-width="1.3" stroke-dasharray="6,6" opacity="0.8"/>',
            f'<circle cx="{x_pos(best_weight):.2f}" cy="{y_pos(best_mean):.2f}" r="5.5" fill="#d62728"/>',
            f'<rect x="{width - 312}" y="{top + 14}" width="270" height="74" rx="6" fill="white" stroke="#ddd"/>',
            f'<line x1="{width - 292}" y1="{top + 38}" x2="{width - 252}" y2="{top + 38}" stroke="#1f77b4" stroke-width="3"/>',
            f'<text x="{width - 242}" y="{top + 43}" font-family="Arial" font-size="13">Mean Total_Time</text>',
            f'<circle cx="{width - 272}" cy="{top + 66}" r="5" fill="#d62728"/>',
            f'<text x="{width - 242}" y="{top + 71}" font-family="Arial" font-size="13">Transition best {safe_feature}: {best_weight:.2f}, {best_mean:.2f}</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pil_curve(output_path: Path, feature: str, rows: list[dict], Image, ImageDraw, ImageFont) -> None:
    width = 1600
    height = 992
    scale = 1.6
    left = int(88 * scale)
    right = int(32 * scale)
    top = int(64 * scale)
    bottom = int(78 * scale)
    plot_width = width - left - right
    plot_height = height - top - bottom
    weights = [row["weight"] for row in rows]
    means = [row["mean_total_time"] for row in rows]
    stds = [row["std_total_time"] for row in rows]
    lower_values = [mean - std for mean, std in zip(means, stds)]
    upper_values = [mean + std for mean, std in zip(means, stds)]
    y_min = math.floor(min(lower_values) - 1.0)
    y_max = math.ceil(max(upper_values) + 1.0)
    marker = _transition_best_point(rows)
    best_weight = marker["weight"]
    best_mean = marker["mean_total_time"]

    def x_pos(weight: float) -> float:
        return left + (weight + 10.0) / 20.0 * plot_width

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    def points_for(values: list[float]) -> list[tuple[int, int]]:
        return [
            (round(x_pos(weight)), round(y_pos(value)))
            for weight, value in zip(weights, values)
        ]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    title_font = ImageFont.load_default()

    title = f"Sensitivity Curve: {feature}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_bbox[2] - title_bbox[0])) / 2, 28), title, fill=(0, 0, 0, 255), font=title_font)
    draw.text((width / 2 - 26, height - 42), "Weight", fill=(0, 0, 0, 255), font=font)
    draw.text((24, height / 2), "Total_Time", fill=(0, 0, 0, 255), font=font)

    x_ticks = [-10, -5, 0, 5, 10]
    y_step = max(1, math.ceil((y_max - y_min) / 8))
    y_start = math.ceil(y_min / y_step) * y_step
    y_ticks = list(range(y_start, y_max + 1, y_step))
    for tick in x_ticks:
        x = round(x_pos(tick))
        draw.line((x, top, x, height - bottom), fill=(230, 230, 230, 255), width=1)
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, height - bottom + 18), label, fill=(0, 0, 0, 255), font=font)
    for tick in y_ticks:
        y = round(y_pos(tick))
        draw.line((left, y, width - right, y), fill=(230, 230, 230, 255), width=1)
        label = f"{tick:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((left - (bbox[2] - bbox[0]) - 12, y - 6), label, fill=(0, 0, 0, 255), font=font)

    draw.rectangle((left, top, width - right, height - bottom), outline=(51, 51, 51, 255), width=2)
    upper_points = points_for(upper_values)
    lower_points = list(reversed(points_for(lower_values)))
    draw.polygon(upper_points + lower_points, fill=(31, 119, 180, 38))
    draw.line(points_for(means), fill=(31, 119, 180, 255), width=4, joint="curve")
    best_x = round(x_pos(best_weight))
    best_y = round(y_pos(best_mean))
    for y in range(top, height - bottom, 18):
        draw.line((best_x, y, best_x, min(y + 9, height - bottom)), fill=(214, 39, 40, 210), width=2)
    draw.ellipse((best_x - 8, best_y - 8, best_x + 8, best_y + 8), fill=(214, 39, 40, 255))

    legend_x = width - 500
    legend_y = top + 24
    draw.rounded_rectangle((legend_x, legend_y, legend_x + 430, legend_y + 118), radius=8, fill=(255, 255, 255, 245), outline=(220, 220, 220, 255))
    draw.line((legend_x + 24, legend_y + 36, legend_x + 88, legend_y + 36), fill=(31, 119, 180, 255), width=4)
    draw.text((legend_x + 104, legend_y + 28), "Mean Total_Time", fill=(0, 0, 0, 255), font=font)
    draw.ellipse((legend_x + 54, legend_y + 75, legend_x + 70, legend_y + 91), fill=(214, 39, 40, 255))
    draw.text((legend_x + 104, legend_y + 72), f"Transition best: {best_weight:.2f}, {best_mean:.2f}", fill=(0, 0, 0, 255), font=font)

    image.save(output_path)


def _transition_best_point(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda row: row["weight"])
    weights = [row["weight"] for row in rows]
    means = [row["mean_total_time"] for row in rows]
    best_mean = min(means)
    tolerance = 0.005
    selected = [
        index
        for index, mean in enumerate(means)
        if mean <= best_mean + tolerance
    ]
    if len(selected) == len(rows):
        neutral_weight = 0.0 if 0.0 in weights else weights[len(weights) // 2]
        neutral_index = weights.index(neutral_weight)
        return {
            "weight": neutral_weight,
            "mean_total_time": means[neutral_index],
        }

    groups = _contiguous_index_groups(selected)
    candidates = []
    last_index = len(rows) - 1
    for group in groups:
        if group[-1] == last_index:
            candidate_index = group[0]
        elif group[0] == 0:
            candidate_index = group[-1]
        else:
            candidate_index = min(group, key=lambda index: abs(weights[index]))
        candidates.append(candidate_index)
    candidate_index = min(candidates, key=lambda index: abs(weights[index]))
    return {
        "weight": weights[candidate_index],
        "mean_total_time": means[candidate_index],
    }


def _contiguous_index_groups(indices: list[int]) -> list[list[int]]:
    groups = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


if __name__ == "__main__":
    main()
