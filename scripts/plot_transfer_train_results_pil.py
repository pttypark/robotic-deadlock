from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_train_400_time_limited"
    / "train_trials_all_policies.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR
    / "final_output"
    / "transfer_generalization_train_400_time_limited"
)

COLORS = {
    "basic_heuristic": (44, 160, 44),
    "bo_heuristic": (31, 119, 180),
    "pso_heuristic": (214, 39, 40),
}
LABELS = {
    "basic_heuristic": "Basic-Heuristic",
    "bo_heuristic": "BO-Heuristic",
    "pso_heuristic": "PSO-Heuristic",
}


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    grouped = group_best_so_far(rows)
    draw_chart(
        grouped,
        args.output_dir / "best_total_time_over_training_time.png",
        "Best Total_Time by Training Time",
    )
    for policy, points in grouped.items():
        draw_chart(
            {policy: points},
            args.output_dir / f"{policy}_best_total_time_over_training_time.png",
            f"{LABELS[policy]} Best Total_Time by Training Time",
        )
    print(f"wrote plots: {args.output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def group_best_so_far(rows: list[dict]) -> dict[str, list[tuple[float, float]]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    points: dict[str, list[tuple[float, float]]] = {}
    for policy, policy_rows in grouped.items():
        policy_rows.sort(key=lambda item: int(item["evaluation_index"]))
        elapsed = 0.0
        best = float("inf")
        policy_points = []
        for row in policy_rows:
            elapsed += float(row["elapsed_seconds"])
            best = min(best, float(row["avg_total_time"]))
            policy_points.append((elapsed / 60.0, best))
        points[policy] = policy_points
    return points


def draw_chart(
    grouped: dict[str, list[tuple[float, float]]],
    path: Path,
    title: str,
) -> None:
    width, height = 1280, 760
    margin_left, margin_right, margin_top, margin_bottom = 110, 42, 82, 96
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    image = Image.new("RGB", (width, height), (250, 251, 252))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    all_x = [x for points in grouped.values() for x, _ in points]
    all_y = [y for points in grouped.values() for _, y in points]
    x_min, x_max = 0.0, max(30.0, max(all_x) if all_x else 30.0)
    y_min = min(all_y) - 0.05 if all_y else 57.0
    y_max = max(all_y) + 0.10 if all_y else 59.0
    if y_max - y_min < 0.4:
        y_mid = (y_min + y_max) / 2
        y_min, y_max = y_mid - 0.25, y_mid + 0.25

    def sx(value: float) -> int:
        return int(margin_left + (value - x_min) / (x_max - x_min) * chart_w)

    def sy(value: float) -> int:
        return int(margin_top + (y_max - value) / (y_max - y_min) * chart_h)

    draw.text((margin_left, 30), title, fill=(24, 30, 36), font=font)
    draw.rectangle(
        (margin_left, margin_top, margin_left + chart_w, margin_top + chart_h),
        outline=(180, 186, 192),
        width=1,
    )

    for i in range(7):
        x_value = x_min + (x_max - x_min) * i / 6
        x = sx(x_value)
        draw.line((x, margin_top, x, margin_top + chart_h), fill=(226, 230, 234))
        draw.text((x - 12, margin_top + chart_h + 14), f"{x_value:.0f}", fill=(65, 72, 80), font=font)

    for i in range(6):
        y_value = y_min + (y_max - y_min) * i / 5
        y = sy(y_value)
        draw.line((margin_left, y, margin_left + chart_w, y), fill=(226, 230, 234))
        draw.text((30, y - 6), f"{y_value:.2f}", fill=(65, 72, 80), font=font)

    x5 = sx(5.0)
    draw.line((x5, margin_top, x5, margin_top + chart_h), fill=(100, 100, 100), width=2)
    draw.text((x5 + 6, margin_top + 8), "5 min checkpoint", fill=(80, 80, 80), font=font)

    for policy in ("basic_heuristic", "bo_heuristic", "pso_heuristic"):
        points = grouped.get(policy, [])
        if not points:
            continue
        color = COLORS[policy]
        screen_points = [(sx(x), sy(y)) for x, y in points]
        if len(screen_points) >= 2:
            draw.line(screen_points, fill=color, width=3)
        for x, y in screen_points[:: max(1, len(screen_points) // 60)]:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
        x, y = screen_points[-1]
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(20, 20, 20))
        draw.text((x + 8, y - 8), f"{points[-1][1]:.2f}", fill=color, font=font)

    legend_x, legend_y = width - 270, 34
    for index, policy in enumerate(("basic_heuristic", "bo_heuristic", "pso_heuristic")):
        y = legend_y + index * 24
        draw.rectangle((legend_x, y, legend_x + 14, y + 14), fill=COLORS[policy])
        draw.text((legend_x + 22, y), LABELS[policy], fill=(35, 40, 48), font=font)

    draw.text((margin_left + chart_w // 2 - 120, height - 45), "Training time within each policy (minutes)", fill=(35, 40, 48), font=font)
    draw.text((16, margin_top + chart_h // 2 - 40), "Best avg Total_Time", fill=(35, 40, 48), font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


if __name__ == "__main__":
    main()
