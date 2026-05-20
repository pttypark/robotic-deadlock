# Total-Time Policy Comparison

## Common setup

| item | value |
|---|---:|
| AGVs | 16 |
| scenarios per cycle | 100 |
| corridor_length | 8 |
| west_exit_extension | 0 |
| admission_window_steps | 2 |
| shared_area_capacity | 2 |
| max_planned_spawn_step | 8 |
| per-direction AGV count range | 2-6 |
| fixed weight sets | 50 |

Each run samples one fixed scenario: NORTH/SOUTH/WEST/EAST AGV counts, per-AGV planned start times, and random end-point assignments. The same scenario is replayed for FCFS, Fixed Heuristic, and BO Heuristic.

One cycle means running the full set of sampled scenarios once. FCFS is executed for one cycle as the deterministic baseline. Fixed Heuristic evaluates the pre-composed weight sets over the same cycle and keeps the lowest-average `total_time` set. BO starts from that Fixed best set and continues optimizing the same five weights.

The only evaluation metric used in summaries is `total_time`.

## Fixed heuristic search

- Fixed candidate sets: `50`
- Fixed greedy best avg total_time: `49.95`
- Fixed greedy best weights: `{"approach_queue": 0.0, "exit_competition": 1.2, "maneuver": 0.0, "path_conflict": 2.4, "remaining_path": 0.0, "same_direction_backlog": 0.0, "waiting": 1.8}`

## Bayesian optimization

The BO stage optimizes the five heuristic weights from the report: `waiting`, `maneuver`, `exit_competition`, `path_conflict`, and `approach_queue`. It uses a lightweight Gaussian Process surrogate and Lower Confidence Bound acquisition, `LCB(w) = mu(w) - kappa * sigma(w)`, to minimize average `total_time` over the same 100-scenario cycle.

- BO trials: `50`
- BO scenarios per trial: `100`
- BO initial local trials: `8`
- BO initial random trials: `8`
- BO exploration mode: `aggressive`
- BO kappa: `3.0`
- BO best training avg total_time: `49.66`

## Summary

| policy | avg total_time | min | max | std | best count | unique best count |
|---|---:|---:|---:|---:|---:|---:|
| fcfs | 54.740 | 42 | 69 | 5.239 | 3 | 1 |
| fixed_heuristic | 49.950 | 38 | 60 | 4.783 | 64 | 21 |
| bo_heuristic | 49.660 | 40 | 66 | 4.789 | 78 | 35 |

## Files

- `experiment_settings.csv`: sampled allocations, start plans, and end plans
- `raw_runs.csv`: one row per policy per run
- `policy_summary.csv`: total_time-only statistics by policy
- `trial_comparison.csv`: per-run winner and pairwise total_time deltas
- `best_policy_by_run.csv`: one row per scenario showing which policy had the lowest total_time
- `fixed_greedy_search.csv`: one row per pre-composed Fixed weight set
- `bo_trials.csv`: BO search trace
- `bo_best_weights.json`: selected BO weights, bounds, and Fixed search result
