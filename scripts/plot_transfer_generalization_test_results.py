from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = PROJECT_DIR / "final_output" / "transfer_generalization_test_results"
DEFAULT_DESKTOP_DIR = Path.home() / "OneDrive" / "바탕 화면" / "transfer_generalization_test_results_plots"

POLICIES = ("fcfs", "basic_5min", "basic_30min", "bo_5min", "bo_30min", "pso_5min", "pso_30min")
LAYOUT_LABELS = {"single_cross": "Single Cross", "double_cross": "Double Cross"}
POLICY_LABELS = {
    "fcfs": "FCFS",
    "basic_5min": "Basic 5m",
    "basic_30min": "Basic 30m",
    "bo_5min": "BO 5m",
    "bo_30min": "BO 30m",
    "pso_5min": "PSO 5m",
    "pso_30min": "PSO 30m",
}
COLORS = {
    "fcfs": (105, 115, 128),
    "basic_5min": (92, 177, 117),
    "basic_30min": (39, 138, 85),
    "bo_5min": (82, 158, 226),
    "bo_30min": (31, 102, 184),
    "pso_5min": (232, 119, 111),
    "pso_30min": (199, 56, 62),
}


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir
    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    overall = read_csv(result_dir / "transfer_test_summary_overall.csv")
    by_agv = read_csv(result_dir / "transfer_test_summary_by_agv.csv")
    by_scenario = read_csv(result_dir / "transfer_test_summary_by_scenario.csv")
    by_scenario_agv = read_csv(result_dir / "transfer_test_summary_by_scenario_agv.csv")

    draw_overall_bars(overall, plot_dir / "01_overall_avg_total_time_by_layout_policy.png")
    draw_agv_lines(by_agv, plot_dir / "02_avg_total_time_by_agv_and_policy.png")
    draw_scenario_bars(by_scenario, plot_dir / "03_avg_total_time_by_scenario_policy.png")
    draw_fcfs_improvement(overall, plot_dir / "04_fcfs_improvement_by_layout_policy.png")
    draw_5min_30min_delta(overall, plot_dir / "05_5min_vs_30min_delta.png")
    draw_scenario_agv_heatmaps(by_scenario_agv, plot_dir)

    if args.copy_to_desktop:
        args.desktop_dir.mkdir(parents=True, exist_ok=True)
        for path in plot_dir.glob("*.png"):
            shutil.copy2(path, args.desktop_dir / path.name)
        print(f"copied plots to desktop: {args.desktop_dir}")
    print(f"wrote plots: {plot_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--copy-to-desktop", action="store_true")
    parser.add_argument("--desktop-dir", type=Path, default=DEFAULT_DESKTOP_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def draw_overall_bars(rows: list[dict], path: Path) -> None:
    for_plot = []
    for layout in ("single_cross", "double_cross"):
        items = [row for row in rows if row["layout_key"] == layout]
        for_plot.append((LAYOUT_LABELS[layout], [(row["policy"], float(row["avg_total_time"])) for row in sort_policy(items)]))
    draw_grouped_bars(
        for_plot,
        path,
        title="Overall Avg Total_Time by Layout and Policy",
        y_label="Avg Total_Time",
        lower_is_better=True,
    )


def draw_agv_lines(rows: list[dict], path: Path) -> None:
    canvas = Canvas(path, "Avg Total_Time by AGV Count and Policy", width=1460, height=760)
    panels = [("single_cross", 82, 104, 20, 28), ("double_cross", 84, 124, 24, 36)]
    for panel_index, (layout, y_min, y_max, x_min, x_max) in enumerate(panels):
        x0 = 92 + panel_index * 680
        y0 = 110
        w = 560
        h = 480
        canvas.panel(x0, y0, w, h, LAYOUT_LABELS[layout])
        layout_rows = [row for row in rows if row["layout_key"] == layout]
        for policy in POLICIES:
            policy_rows = sorted([row for row in layout_rows if row["policy"] == policy], key=lambda row: int(row["total_agvs"]))
            pts = []
            for row in policy_rows:
                x = scale(float(row["total_agvs"]), x_min, x_max, x0 + 52, x0 + w - 36)
                y = scale(float(row["avg_total_time"]), y_min, y_max, y0 + h - 52, y0 + 42)
                pts.append((x, y))
            canvas.polyline(pts, COLORS[policy], width=3)
            for x, y in pts:
                canvas.dot(x, y, COLORS[policy])
        canvas.axes(x0, y0, w, h, x_label="AGV count", y_label="Avg Total_Time")
        for value in range(x_min, x_max + 1, 4 if layout == "single_cross" else 6):
            x = scale(value, x_min, x_max, x0 + 52, x0 + w - 36)
            canvas.text((x - 8, y0 + h - 28), str(value), fill=(60, 66, 74))
        for value in range(int(y_min), int(y_max) + 1, 5 if layout == "single_cross" else 10):
            y = scale(value, y_min, y_max, y0 + h - 52, y0 + 42)
            canvas.text((x0 + 14, y - 6), str(value), fill=(60, 66, 74))
    canvas.legend(POLICIES, POLICY_LABELS, COLORS, x=100, y=645)
    canvas.save()


def draw_scenario_bars(rows: list[dict], path: Path) -> None:
    groups = []
    for layout in ("single_cross", "double_cross"):
        for case_id in ("5", "6", "7", "8"):
            items = [
                row
                for row in rows
                if row["layout_key"] == layout and row["scenario_case_id"] == case_id
            ]
            label = f"{LAYOUT_LABELS[layout]} S{case_id}"
            groups.append((label, [(row["policy"], float(row["avg_total_time"])) for row in sort_policy(items)]))
    draw_grouped_bars(
        groups,
        path,
        title="Avg Total_Time by Scenario Case and Policy",
        y_label="Avg Total_Time",
        lower_is_better=True,
        width=1760,
        height=860,
    )


def draw_fcfs_improvement(rows: list[dict], path: Path) -> None:
    groups = []
    for layout in ("single_cross", "double_cross"):
        layout_rows = {row["policy"]: float(row["avg_total_time"]) for row in rows if row["layout_key"] == layout}
        fcfs = layout_rows["fcfs"]
        values = []
        for policy in POLICIES[1:]:
            values.append((policy, fcfs - layout_rows[policy]))
        groups.append((LAYOUT_LABELS[layout], values))
    draw_grouped_bars(
        groups,
        path,
        title="Improvement vs FCFS: Avg Total_Time Reduction",
        y_label="FCFS avg - policy avg",
        lower_is_better=False,
    )


def draw_5min_30min_delta(rows: list[dict], path: Path) -> None:
    groups = []
    pairs = [("basic", "basic_5min", "basic_30min"), ("bo", "bo_5min", "bo_30min"), ("pso", "pso_5min", "pso_30min")]
    for layout in ("single_cross", "double_cross"):
        layout_rows = {row["policy"]: float(row["avg_total_time"]) for row in rows if row["layout_key"] == layout}
        values = [(name, layout_rows[p30] - layout_rows[p5]) for name, p5, p30 in pairs]
        groups.append((LAYOUT_LABELS[layout], values))
    colors = {"basic": COLORS["basic_30min"], "bo": COLORS["bo_30min"], "pso": COLORS["pso_30min"]}
    labels = {"basic": "Basic 30m-5m", "bo": "BO 30m-5m", "pso": "PSO 30m-5m"}
    draw_grouped_bars(
        groups,
        path,
        title="30min - 5min Avg Total_Time Delta",
        y_label="Delta Avg Total_Time",
        lower_is_better=True,
        policy_order=("basic", "bo", "pso"),
        colors=colors,
        labels=labels,
    )


def draw_scenario_agv_heatmaps(rows: list[dict], plot_dir: Path) -> None:
    key_policies = ("fcfs", "basic_5min", "basic_30min", "bo_5min", "bo_30min", "pso_5min", "pso_30min")
    for layout in ("single_cross", "double_cross"):
        layout_rows = [row for row in rows if row["layout_key"] == layout]
        agv_counts = sorted({int(row["total_agvs"]) for row in layout_rows})
        cases = ("5", "6", "7", "8")
        canvas = Canvas(plot_dir / f"06_{layout}_scenario_agv_policy_heatmap.png", f"{LAYOUT_LABELS[layout]} Avg Total_Time Heatmap", width=1620, height=820)
        cell_w, cell_h = 130, 48
        x0, y0 = 190, 118
        values = [float(row["avg_total_time"]) for row in layout_rows]
        v_min, v_max = min(values), max(values)
        for ai, agv in enumerate(agv_counts):
            for ci, case_id in enumerate(cases):
                block_y = y0 + (ai * len(cases) + ci) * cell_h
                canvas.text((34, block_y + 15), f"AGV {agv} / S{case_id}", fill=(35, 40, 48))
                for pi, policy in enumerate(key_policies):
                    row = next(
                        item for item in layout_rows
                        if int(item["total_agvs"]) == agv and item["scenario_case_id"] == case_id and item["policy"] == policy
                    )
                    value = float(row["avg_total_time"])
                    color = heat_color(value, v_min, v_max)
                    x = x0 + pi * cell_w
                    canvas.draw.rectangle((x, block_y, x + cell_w - 4, block_y + cell_h - 4), fill=color, outline=(235, 238, 240))
                    canvas.text((x + 36, block_y + 15), f"{value:.1f}", fill=(25, 29, 34))
        for pi, policy in enumerate(key_policies):
            canvas.text((x0 + pi * cell_w + 12, 82), POLICY_LABELS[policy], fill=(35, 40, 48))
        canvas.save()


def draw_grouped_bars(
    groups,
    path: Path,
    title: str,
    y_label: str,
    lower_is_better: bool,
    width: int = 1380,
    height: int = 760,
    policy_order=POLICIES,
    colors=COLORS,
    labels=POLICY_LABELS,
) -> None:
    canvas = Canvas(path, title, width=width, height=height)
    x0, y0 = 90, 100
    chart_w, chart_h = width - 150, height - 220
    all_values = [value for _, items in groups for _, value in items]
    y_min = min(0.0, min(all_values) - 0.5) if not lower_is_better else min(all_values) - 2
    y_max = max(all_values) + 2
    if y_max - y_min < 1:
        y_max += 1
        y_min -= 1
    canvas.axes(x0, y0, chart_w, chart_h, x_label="", y_label=y_label)
    group_w = chart_w / max(1, len(groups))
    bar_w = min(22, group_w / (len(policy_order) + 2))
    zero_y = scale(0.0, y_min, y_max, y0 + chart_h, y0)
    for gi, (group_label, items) in enumerate(groups):
        values_by_policy = dict(items)
        gx = x0 + gi * group_w + group_w * 0.12
        for pi, policy in enumerate(policy_order):
            if policy not in values_by_policy:
                continue
            value = values_by_policy[policy]
            x = int(gx + pi * bar_w * 1.15)
            y = scale(value, y_min, y_max, y0 + chart_h, y0)
            y_base = zero_y if y_min < 0 < y_max else y0 + chart_h
            canvas.draw.rectangle((x, min(y, y_base), x + bar_w, max(y, y_base)), fill=colors[policy])
        canvas.text((int(x0 + gi * group_w + 6), y0 + chart_h + 18), group_label, fill=(35, 40, 48))
    for tick in nice_ticks(y_min, y_max, 6):
        y = scale(tick, y_min, y_max, y0 + chart_h, y0)
        canvas.draw.line((x0, y, x0 + chart_w, y), fill=(228, 232, 236))
        canvas.text((28, y - 6), f"{tick:.1f}", fill=(60, 66, 74))
    canvas.legend(policy_order, labels, colors, x=90, y=height - 82)
    canvas.save()


class Canvas:
    def __init__(self, path: Path, title: str, width: int = 1280, height: int = 760) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.image = Image.new("RGB", (width, height), (250, 251, 252))
        self.draw = ImageDraw.Draw(self.image)
        self.font = ImageFont.load_default()
        self.title(title)

    def title(self, title: str) -> None:
        self.text((34, 28), title, fill=(24, 30, 36))

    def text(self, xy, text: str, fill=(0, 0, 0)) -> None:
        self.draw.text(xy, text, fill=fill, font=self.font)

    def axes(self, x: int, y: int, w: int, h: int, x_label: str, y_label: str) -> None:
        self.draw.rectangle((x, y, x + w, y + h), outline=(176, 184, 192), width=1)
        if x_label:
            self.text((x + w // 2 - 40, y + h + 44), x_label, fill=(35, 40, 48))
        if y_label:
            self.text((x - 70, y + 4), y_label, fill=(35, 40, 48))

    def panel(self, x: int, y: int, w: int, h: int, label: str) -> None:
        self.axes(x, y, w, h, "", "Avg Total_Time")
        self.text((x + 12, y + 12), label, fill=(24, 30, 36))

    def polyline(self, points, color, width=2) -> None:
        if len(points) >= 2:
            self.draw.line(points, fill=color, width=width)

    def dot(self, x: int, y: int, color) -> None:
        self.draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(30, 34, 38))

    def legend(self, policies, labels, colors, x: int, y: int) -> None:
        cx, cy = x, y
        for policy in policies:
            self.draw.rectangle((cx, cy, cx + 14, cy + 14), fill=colors[policy])
            self.text((cx + 20, cy), labels[policy], fill=(35, 40, 48))
            cx += 128
            if cx > self.width - 170:
                cx = x
                cy += 24

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(self.path)


def sort_policy(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: POLICIES.index(row["policy"]))


def scale(value: float, v_min: float, v_max: float, out_min: float, out_max: float) -> int:
    if v_max == v_min:
        return int((out_min + out_max) / 2)
    return int(out_min + (value - v_min) / (v_max - v_min) * (out_max - out_min))


def nice_ticks(v_min: float, v_max: float, count: int) -> list[float]:
    if count <= 1:
        return [v_min]
    return [v_min + (v_max - v_min) * index / (count - 1) for index in range(count)]


def heat_color(value: float, v_min: float, v_max: float) -> tuple[int, int, int]:
    ratio = 0.0 if v_max == v_min else (value - v_min) / (v_max - v_min)
    good = (190, 231, 204)
    bad = (244, 182, 174)
    return tuple(int(good[i] * (1 - ratio) + bad[i] * ratio) for i in range(3))


if __name__ == "__main__":
    main()
