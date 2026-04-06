# RL Deadlock Guide

This repository now contains the minimum hooks needed to study deadlock-aware AMR control with reinforcement learning.

## Priority Parameters

- `n_agents`: primary source of congestion and multi-agent coordination difficulty.
- `sensor_range`: determines how early a robot can react to traffic and humans.
- `request_queue_size`: increases traffic density and shelf contention.
- `layout`: controls bottlenecks, alternate routes, and corridor width.
- `human_count`: number of dynamic human obstacles.
- `human_safety_radius`: hard safety buffer around each human.
- `human_move_prob`: probability that a human moves on each step.
- `human_wait_prob`: probability that a human unexpectedly stops.
- `deadlock_patience`: how many unchanged steps trigger deadlock detection.
- `reward_shaping`: optional RL reward weights for waiting, deadlock, human blocking, and progress.

## RL Signals Available In `info`

`Warehouse.step()` now exposes:

- `info["metrics"]["throughput"]`
- `info["metrics"]["stalled_agents"]`
- `info["metrics"]["avg_wait_steps"]`
- `info["metrics"]["blocked_by_human"]`
- `info["metrics"]["blocked_by_agent"]`
- `info["metrics"]["deadlock_active"]`
- `info["metrics"]["deadlock_agents"]`
- `info["blockers"]`
- `info["humans"]`
- `info["wait_steps"]`

These are intended for reward design, debugging, and offline dataset generation.

## Recommended Research Progression

1. Train without humans on `single_lane` and `intersection`.
2. Move to `factory_floor` for simple multi-intersection traffic.
3. Move to `factory_complex` for timed aisle closures and multi-zone circulation.
4. Add reward shaping for wait penalties and progress bonuses.
5. Train with `human_blockage`, `factory_floor`, and `factory_complex` using multiple patrol routes.
6. Increase `human_count`, route overlap, and stop probability.
7. Move to learned human motion models or replayed human traces.

## Visualization

The renderer now shows:

- humans as blue discs
- human danger zones as red overlays
- blocked action links
- deadlocked agents highlighted in gold
- a HUD with throughput, wait, and blocking metrics

Run:

```bash
python main.py --scenario human_blockage --render
python main.py --scenario factory_floor --render --hold
python main.py --scenario factory_floor --save-dir outputs --save-name factory_demo
python main.py --scenario factory_complex --render --hold
python main.py --scenario factory_complex --save-dir outputs --save-name factory_complex_demo
```

Saved artifacts:

- `*.gif` when Pillow is installed
- `*.npz` fallback if Pillow is missing
- `*.json` rollout metadata and summary stats

## Next Useful Extensions

- add a dedicated image layer for blocker identity and queue pressure
- export episode logs to JSON for offline RL or behavior cloning
- replace simple human motion with trajectory files or crowd simulators
- add centralized traffic-controller actions alongside robot actions
- add a hierarchical controller that selects intersection reservations before low-level AMR actions
- replace the waypoint baseline in `factory_complex` with a learned dispatcher or MAPPO policy
