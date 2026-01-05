# A2C Atari — Evaluation + TensorBoard logging

import gymnasium as gym
import torch
import numpy as np
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

RUN_DIR = "runs/ALE/Breakout-v5__a2c_atari__1__1767432055"
MODEL_PATH = f"{RUN_DIR}/a2c_model.pt"
EVAL_EPISODES = 10
VIDEO_DIR = "videos/a2c_eval_fixed"


def make_env():
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        VIDEO_DIR,
        episode_trigger=lambda i: True,
        video_length=0,
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)

    env = NoopResetEnv(env, 30)
    env = MaxAndSkipEnv(env, 4)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    return env


class ActorCritic(torch.nn.Module):
    def __init__(self, env):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
        )

        self.policy = torch.nn.Linear(512, env.action_space.n)
        self.value = torch.nn.Linear(512, 1)

    def get_action(self, x):
        hidden = self.net(x / 255.0)
        logits = self.policy(hidden)
        probs = Categorical(logits=logits)
        return probs.sample()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env()
    agent = ActorCritic(env).to(device)

    print(f"Loading model: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    agent.load_state_dict(state)

    # 👉 TensorBoard writer (writes into SAME TRAINING RUN)
    writer = SummaryWriter(RUN_DIR)

    returns = []

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        obs = torch.tensor(obs, device=device)

        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                action = agent.get_action(obs.unsqueeze(0))

            obs, r, term, trunc, _ = env.step(action.item())
            obs = torch.tensor(obs, device=device)
            ep_ret += r
            done = term or trunc

        print(f"Episode {ep}: return={ep_ret}")

        returns.append(ep_ret)

        # 👉 log this episode to TensorBoard
        writer.add_scalar("eval/episodic_return", ep_ret, ep)

    env.close()
    writer.close()

    print("\n==== A2C EVALUATION (FULL GAME) ====")
    print(f"mean: {np.mean(returns):.2f}")
    print(f"min : {np.min(returns)}")
    print(f"max : {np.max(returns)}")
    print(f"std : {np.std(returns):.2f}")
