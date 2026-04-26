# Research Results Sync

PDF direction: shared-area approach/core/purpose-area를 기준으로 후보 AMR을 그래프로 구성하고, action masking 후 진입/대기를 결정한다.

## Code Sync

- Shared-area: `deadlock.dsd_fsd.shared_area_plans`에서 corridor/hotspot 병목을 같은 초기 조건으로 정의한다.
- Approach/Core/Purpose: `LocalGraphAdmissionController`가 aisle 양 끝 접근 구간과 protected conflict core를 분리해 진입 여부를 판단한다.
- Graph model: 후보 AMR 노드에 누적 대기, core 내부 여부, 출구 가용성, 반대 방향 경쟁, 같은 방향 대기열, 목표 거리 feature를 부여한다.
- Action masking: conflict core 점유 중 또는 더 높은 score 후보가 있으면 해당 AMR의 `ENTER`를 `WAIT`로 마스킹한다.
- KPI: completed, Favg, Fmax, max_active_wait, graph_decisions, admission_holds, max_queue_length를 결과 지표로 출력한다.

## 500-Step Result

| Scenario | Completed | Favg | Fmax | Max active wait | Graph decisions | Admission holds | Max queue |
|---|---:|---:|---:|---:|---:|---:|---:|
| Rule-based shared-cross | 0 | 0.00 | 0 | 493 | 0 | 0 | 0 |
| OUR shared-cross | 0 | 0.00 | 0 | 2 | 499 | 748 | 4 |
| Rule-based shared-queue | 1 | 20.00 | 20 | 491 | 0 | 0 | 0 |
| OUR shared-queue | 1 | 20.00 | 20 | 2 | 337 | 749 | 4 |

## Slide Insert

연구 결과 요약

- Rule-based shared-cross는 반대 방향 AMR이 protected core에 동시에 진입하면서 completed=0으로 교착이 지속됨.
- OUR(GNN-style local admission)는 conflict core 점유 여부, 출구 가용성, 접근 대기열, 반대 방향 경쟁 관계를 그래프 점수에 반영해 한 번에 1대만 진입시킴.
- 현재 prototype의 정량 결과는 처리량 향상보다는 max active wait 감소에 초점이 있음. Shared-cross는 493에서 2, shared-queue는 491에서 2로 줄어듦.
- 즉, 현 단계 코드는 학습 완료 모델이 아니라 shared-area 진입을 제한해 교착 지속 시간을 줄이는 구조 검증용 baseline임.
- 따라서 본 프로젝트의 초점은 전역 경로 재계획이 아니라 shared-area 주변 후보 AMR의 국소 진입 순서 조정으로 정리할 수 있음.

발표용 한 줄 결론

> Shared-area에서 발생하는 교착은 단순 rule-base 양보 규칙만으로는 지속되지만, GNN-style 국소 진입 조정은 후보 AMR 간 관계를 반영해 진입 순서를 제한함으로써 병목 흐름을 회복했다.

Generated videos

- `C:\Users\승승협\Desktop\shared_area_videos\02_shared_cross_deadlock_fsd.mp4`
- `C:\Users\승승협\Desktop\shared_area_videos\03_shared_queue_deadlock_fsd.mp4`
- `C:\Users\승승협\Desktop\shared_area_videos\04_shared_cross_gnn_admission.mp4`
- `C:\Users\승승협\Desktop\shared_area_videos\05_shared_queue_gnn_admission.mp4`
