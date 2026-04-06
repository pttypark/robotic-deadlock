import argparse
import os
import sys

import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:  # pragma: no cover
    PPO = None


def flatten_tuple_observation(observation):
    return np.concatenate([np.asarray(x).flatten() for x in observation], axis=0).astype(np.float32)


class FlattenTupleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Tuple):
            raise ValueError("Expected Tuple observation space")

        lows = [space.low.flatten() for space in env.observation_space]
        highs = [space.high.flatten() for space in env.observation_space]
        low = np.concatenate(lows, axis=0).astype(np.float32)
        high = np.concatenate(highs, axis=0).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return flatten_tuple_observation(observation)


class JointActionDiscrete(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Tuple):
            raise ValueError("Expected Tuple action space")
        self._action_sizes = [space.n for space in env.action_space]
        self._size = int(np.prod(self._action_sizes, dtype=np.int64))
        if self._size > 65536:
            raise ValueError(
                f"Joint action space too large: {self._size}. Use fewer agents or smaller action spaces."
            )
        self.action_space = gym.spaces.Discrete(self._size)

    def action(self, action):
        action = int(action)
        joint_action = []
        for size in reversed(self._action_sizes):
            joint_action.append(action % size)
            action //= size
        return tuple(reversed(joint_action))


def make_env(env_id):
    env = gym.make(env_id)
    env = JointActionDiscrete(env)
    env = FlattenTupleObservation(env)
    return env


def train(args):
    if PPO is None:
        print("stable-baselines3가 설치되어 있지 않습니다. 다음 명령을 실행하세요:")
        print("python -m pip install stable-baselines3 torch")
        sys.exit(1)

    env = DummyVecEnv([lambda: make_env(args.env)])
    model = PPO("MlpPolicy", env, verbose=1)

    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=save_dir,
        name_prefix="ppo_rware",
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback)
    model_path = os.path.join(save_dir, args.model_name)
    model.save(model_path)
    print(f"Saved trained model to: {model_path}")

    if args.eval_episodes > 0:
        evaluate(model, args.env, args.eval_episodes, render=args.render)


def evaluate(model, env_id, episodes, render=False):
    env = make_env(env_id)
    for ep in range(1, episodes + 1):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            done = terminated or truncated
            if render:
                env.render()
        print(f"Episode {ep}: reward={total_reward:.2f}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PPO agent on RWARE.")
    parser.add_argument("--env", default="rware-tiny-2ag-v2", help="Gymnasium environment ID")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training steps")
    parser.add_argument("--save-dir", default="models", help="Directory to save models")
    parser.add_argument("--model-name", default="ppo_rware", help="Saved model file base name")
    parser.add_argument("--checkpoint-freq", type=int, default=10000, help="Checkpoint save frequency")
    parser.add_argument("--eval-episodes", type=int, default=0, help="Evaluate after training")
    parser.add_argument("--render", action="store_true", help="Render evaluation episodes")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
