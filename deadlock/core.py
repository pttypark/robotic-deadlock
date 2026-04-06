from collections import deque
import json
from pathlib import Path
import time

from deadlock.navigation import FORWARD, NOOP, do_step, rev_dir, turn_seq
import numpy as np


class DeadlockDetector:
    """Detect deadlock when all agent positions remain unchanged for N steps."""

    def __init__(self, patience=3):
        self.patience = patience
        self.history = deque(maxlen=patience + 1)

    def update(self, positions):
        self.history.append(tuple(map(tuple, positions)))
        if len(self.history) < self.patience + 1:
            return False
        return all(state == self.history[0] for state in self.history)

    def reset(self):
        self.history.clear()


class EpisodeRecorder:
    """Collect rendered frames and write them to disk after rollout."""

    def __init__(self, save_dir=None, save_name="rollout", fps=6):
        self.enabled = bool(save_dir)
        self.fps = fps
        self.save_name = save_name
        self.save_dir = Path(save_dir) if save_dir else None
        self.frames = []

    def capture(self, env):
        if not self.enabled:
            return
        if not getattr(env.unwrapped, "renderer", None):
            env.render()
        frame = env.unwrapped.renderer.render(env.unwrapped, return_rgb_array=True)
        self.frames.append(frame)

    def save(self, stats=None):
        if not self.enabled or not self.frames:
            return None

        self.save_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}
        metadata_path = self.save_dir / f"{self.save_name}.json"
        metadata = {"fps": self.fps, "frame_count": len(self.frames), "stats": stats or {}}
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        outputs["json"] = str(metadata_path)

        try:
            from PIL import Image

            images = [Image.fromarray(frame) for frame in self.frames]
            gif_path = self.save_dir / f"{self.save_name}.gif"
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=max(1, int(1000 / max(1, self.fps))),
                loop=0,
            )
            outputs["gif"] = str(gif_path)
        except ImportError:
            npz_path = self.save_dir / f"{self.save_name}.npz"
            np.savez_compressed(npz_path, frames=np.stack(self.frames))
            outputs["npz"] = str(npz_path)

        return outputs


def run_loop(
    env,
    detector,
    resolve_fn,
    max_steps,
    verbose,
    render=False,
    recorder=None,
):
    """Shared scenario rollout loop with optional rendering and metrics printing."""

    n_agents = env.unwrapped.n_agents
    stats = {
        "deadlock_step": None,
        "resolved": False,
        "human_blocks": 0,
        "agent_blocks": 0,
    }

    for step in range(max_steps):
        agents = env.unwrapped.agents
        positions = [(agent.x, agent.y) for agent in agents]

        if verbose:
            directions = [agent.dir.name for agent in agents]
            print(f"  step {step:2d} | pos={positions} dir={directions}")

        if detector.update(positions) and not stats["resolved"]:
            stats["deadlock_step"] = step
            if verbose:
                print(f"\n  DEADLOCK detected at step {step}")
            resolve_fn(env, verbose=verbose, render=render)
            stats["resolved"] = True
            detector.reset()

        if recorder and recorder.enabled:
            recorder.capture(env)
        elif render:
            env.render()

        done = do_step(env, [FORWARD] * n_agents)
        info = env.unwrapped._get_info()
        metrics = info.get("metrics", {})
        stats["human_blocks"] = max(
            stats["human_blocks"], metrics.get("blocked_by_human", 0)
        )
        stats["agent_blocks"] = max(
            stats["agent_blocks"], metrics.get("blocked_by_agent", 0)
        )
        if done:
            break

    return stats


def resolve_single_lane(env, verbose=True, render=False):
    """Priority-based yielding for narrow corridor deadlock."""

    r0_dir = env.unwrapped.agents[0].dir.name
    turn_back = turn_seq(r0_dir, rev_dir(r0_dir))
    turn_fwd = turn_seq(rev_dir(r0_dir), r0_dir)
    r0_seq = turn_back + [FORWARD] * 5 + turn_fwd
    r1_seq = [FORWARD] * len(r0_seq)

    for r0_action, r1_action in zip(r0_seq, r1_seq):
        do_step(env, [r0_action, r1_action])
        if render:
            env.render()
        if verbose:
            agents = env.unwrapped.agents
            print(
                f"    resolving | pos={[(a.x, a.y) for a in agents]} "
                f"dir={[a.dir.name for a in agents]}"
            )


def resolve_intersection(env, verbose=True, render=False):
    """Release one agent to break a four-way cycle."""

    n_agents = env.unwrapped.n_agents
    r0_dir = env.unwrapped.agents[0].dir.name
    turn_actions = turn_seq(r0_dir, rev_dir(r0_dir))
    r0_seq = turn_actions + [FORWARD] * 2
    other_seq = [NOOP] * len(turn_actions) + [FORWARD] * 2

    for idx in range(len(r0_seq)):
        do_step(env, [r0_seq[idx]] + [other_seq[idx]] * (n_agents - 1))
        if render:
            env.render()
        if verbose:
            agents = env.unwrapped.agents
            print(
                f"    resolving | pos={[(a.x, a.y) for a in agents]} "
                f"dir={[a.dir.name for a in agents]}"
            )


def resolve_human_blockage(env, verbose=True, render=False):
    """Simple fallback: all robots pause while a human clears the bottleneck."""

    n_agents = env.unwrapped.n_agents
    for _ in range(4):
        do_step(env, [NOOP] * n_agents)
        if render:
            env.render()
        if verbose:
            info = env.unwrapped._get_info()
            metrics = info.get("metrics", {})
            print(
                "    yielding to human | "
                f"blocked_h={metrics.get('blocked_by_human', 0)} "
                f"deadlock={metrics.get('deadlock_active', False)}"
            )


def hold_render_window(env, enabled=False, poll_interval=0.05):
    """Keep the render window open until the user closes it."""

    if not enabled:
        return

    renderer = getattr(env.unwrapped, "renderer", None)
    if renderer is None:
        env.render()
        renderer = getattr(env.unwrapped, "renderer", None)
    if renderer is None:
        return

    while renderer.isopen:
        env.render()
        time.sleep(poll_interval)
