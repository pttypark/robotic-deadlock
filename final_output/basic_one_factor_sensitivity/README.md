# Basic One-Factor Sensitivity Analysis

## Setup

- Scenario file: `final_output\basic_one_factor_sensitivity\train_scenarios_400.csv`
- Scenario cases: `[1, 2, 3, 4]`
- Scenarios per case: `{1: 100, 2: 100, 3: 100, 4: 100}`
- Total scenarios: `400`
- AGVs: `20`
- Features swept: `['waiting', 'maneuver', 'exit_competition', 'path_conflict', 'approach_queue']`
- Sweep values: `-10.0` to `10.0` by `0.25`
- Controlled feature value: `1.0`
- Selected range tolerance: `0.0%`

For each feature, the other four heuristic parameters are fixed at `1.0`. Only the target feature changes across the sweep grid. FCFS is excluded because it has no weight vector, and PSO/BO are not run during this sensitivity stage.

## Best Weight By Feature

| feature | best weight | avg total_time | selected lower | selected upper |
|---|---:|---:|---:|---:|
| waiting | 10.0 | 58.328 | 10.0 | 10.0 |
| maneuver | -1.75 | 58.430 | -1.75 | -0.25 |
| exit_competition | -0.25 | 58.752 | -0.25 | -0.25 |
| path_conflict | 1.0 | 58.873 | 1.0 | 1.0 |
| approach_queue | -10.0 | 58.873 | -10.0 | 10.0 |

## Files

- `train_scenarios_400.csv`: Scenario 1-4 training set reused by later Train runs
- `sensitivity_trials.csv`: full one-factor sweep curve
- `sensitivity_raw_runs.csv`: one row per feature, weight, and scenario episode
- `sensitivity_summary.csv`: best/worst value per parameter
- `sensitivity_selected_ranges.csv`: range selected from near-best sweep values
