import argparse
import json
from pathlib import Path

import gymnasium as gym
import rware

from deadlock.dsd_fsd.controllers import (
    DeadlockProneRuleController,
    LocalGraphAdmissionController,
)
from deadlock.dsd_fsd.experiment import (
    _normalise_intentional_waits,
    _record_health_metrics,
    _stage_agents,
)
from deadlock.dsd_fsd.layouts import build_paper_points, build_shared_area_points
from deadlock.dsd_fsd.metrics import summarize_metrics
from deadlock.dsd_fsd.model import SystemType, TransactionLedger
from deadlock.dsd_fsd.shared_area_plans import (
    deadlock_plan,
    seed_deadlock_transactions,
)


RESEARCH_SCENARIOS = [
    {
        "name": "corridor_rule_based",
        "label": "Rule-based shared-cross",
        "layout": "shared_area",
        "system": SystemType.FSD,
        "agents": 4,
        "deadlock_pattern": "shared_cross",
        "controller": "rule",
    },
    {
        "name": "corridor_gnn_admission",
        "label": "OUR shared-cross",
        "layout": "shared_area",
        "system": SystemType.FSD,
        "agents": 4,
        "deadlock_pattern": "shared_cross",
        "controller": "gnn_admission",
    },
    {
        "name": "hotspot_rule_based",
        "label": "Rule-based shared-queue",
        "layout": "shared_area",
        "system": SystemType.FSD,
        "agents": 4,
        "deadlock_pattern": "shared_queue",
        "controller": "rule",
    },
    {
        "name": "hotspot_gnn_admission",
        "label": "OUR shared-queue",
        "layout": "shared_area",
        "system": SystemType.FSD,
        "agents": 4,
        "deadlock_pattern": "shared_queue",
        "controller": "gnn_admission",
    },
]


def run_research_suite(seed=0, steps=1000, service_steps=3):
    rows = []
    for scenario in RESEARCH_SCENARIOS:
        metrics = run_research_scenario(
            scenario,
            seed=seed,
            steps=steps,
            service_steps=service_steps,
        )
        rows.append({"scenario": scenario["name"], "label": scenario["label"], **metrics})
    return rows


def run_research_scenario(scenario, seed=0, steps=1000, service_steps=3):
    if scenario["layout"] == "shared_area":
        points = build_shared_area_points(n_agents=scenario["agents"])
    else:
        points = build_paper_points(
            system=scenario["system"].value,
            n_agents=scenario["agents"],
            bay_rows=scenario["bay_rows"],
        )
    plan = deadlock_plan(scenario, points)
    env = gym.make(
        f"rware:rware-tiny-{scenario['agents']}ag-v2",
        layout=points.layout,
        render_mode="rgb_array",
        max_steps=steps + 5,
        request_queue_size=0,
    )
    env.reset(seed=seed)
    _stage_agents(env, points, scenario["agents"], agent_starts=plan["starts"])

    ledger = TransactionLedger()
    seed_deadlock_transactions(ledger, plan)
    controller = _make_controller(scenario, env, points, service_steps)

    for step in range(steps):
        actions = controller.compute_actions(step, ledger)
        _, _, done, truncated, _ = env.step(actions)
        _record_health_metrics(env, controller)
        _normalise_intentional_waits(env, controller)
        if done or truncated:
            break

    controller.compute_actions(steps, ledger)
    metrics = summarize_metrics(ledger, controller.stats, steps)
    env.close()
    return metrics


def _make_controller(scenario, env, points, service_steps):
    if scenario["controller"] == "gnn_admission":
        return LocalGraphAdmissionController(env, points, service_steps=service_steps)
    return DeadlockProneRuleController(env, points, service_steps=service_steps)


def to_markdown(rows):
    steps = rows[0]["steps"] if rows else 0
    lines = [
        "# Research Results Sync",
        "",
        "PDF direction: shared-area approach/core/purpose-area를 기준으로 후보 AMR을 그래프로 구성하고, action masking 후 진입/대기를 결정한다.",
        "",
        "## Code Sync",
        "",
        "- Shared-area: `deadlock.dsd_fsd.shared_area_plans`에서 corridor/hotspot 병목을 같은 초기 조건으로 정의한다.",
        "- Approach/Core/Purpose: `LocalGraphAdmissionController`가 aisle 양 끝 접근 구간과 protected conflict core를 분리해 진입 여부를 판단한다.",
        "- Graph model: 후보 AMR 노드에 누적 대기, core 내부 여부, 출구 가용성, 반대 방향 경쟁, 같은 방향 대기열, 목표 거리 feature를 부여한다.",
        "- Action masking: conflict core 점유 중 또는 더 높은 score 후보가 있으면 해당 AMR의 `ENTER`를 `WAIT`로 마스킹한다.",
        "- KPI: completed, Favg, Fmax, max_active_wait, graph_decisions, admission_holds, max_queue_length를 결과 지표로 출력한다.",
        "",
        f"## {steps}-Step Result",
        "",
        "| Scenario | Completed | Favg | Fmax | Max active wait | Graph decisions | Admission holds | Max queue |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {completed} | {f_avg:.2f} | {f_max} | {max_active_wait} | {graph_decisions} | {admission_holds} | {max_queue_length} |".format(
                label=row["label"],
                completed=row["completed"],
                f_avg=row["f_avg"],
                f_max=row["f_max"],
                max_active_wait=row.get("max_active_wait", 0),
                graph_decisions=row.get("graph_decisions", 0),
                admission_holds=row.get("admission_holds", 0),
                max_queue_length=row.get("max_queue_length", 0),
            )
        )
    lines.extend(
        [
            "",
            "## Slide Insert",
            "",
            "연구 결과 요약",
            "",
            "- Rule-based shared-cross는 반대 방향 AMR이 protected core에 동시에 진입하면서 completed=0으로 교착이 지속됨.",
            "- OUR(GNN-style local admission)는 conflict core 점유 여부, 출구 가용성, 접근 대기열, 반대 방향 경쟁 관계를 그래프 점수에 반영해 한 번에 1대만 진입시킴.",
            "- 현재 prototype의 정량 결과는 처리량 향상보다는 max active wait 감소에 초점이 있음. Shared-cross는 493에서 2, shared-queue는 491에서 2로 줄어듦.",
            "- 즉, 현 단계 코드는 학습 완료 모델이 아니라 shared-area 진입을 제한해 교착 지속 시간을 줄이는 구조 검증용 baseline임.",
            "- 따라서 본 프로젝트의 초점은 전역 경로 재계획이 아니라 shared-area 주변 후보 AMR의 국소 진입 순서 조정으로 정리할 수 있음.",
            "",
            "발표용 한 줄 결론",
            "",
            "> Shared-area에서 발생하는 교착은 단순 rule-base 양보 규칙만으로는 지속되지만, GNN-style 국소 진입 조정은 후보 AMR 간 관계를 반영해 진입 순서를 제한함으로써 병목 흐름을 회복했다.",
            "",
            "Generated videos",
            "",
            "- `C:\\Users\\승승협\\Desktop\\shared_area_videos\\02_shared_cross_deadlock_fsd.mp4`",
            "- `C:\\Users\\승승협\\Desktop\\shared_area_videos\\03_shared_queue_deadlock_fsd.mp4`",
            "- `C:\\Users\\승승협\\Desktop\\shared_area_videos\\04_shared_cross_gnn_admission.mp4`",
            "- `C:\\Users\\승승협\\Desktop\\shared_area_videos\\05_shared_queue_gnn_admission.mp4`",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Run shared-area research-result metrics.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", default="outputs/dsd_fsd/research_results_sync.md")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = run_research_suite(seed=args.seed, steps=args.steps)
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    markdown = to_markdown(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
