# # PPO Atari — Evaluation (9 episodes, full-game)

# import gymnasium as gym
# import torch
# import numpy as np
# from torch.distributions.categorical import Categorical
# from torch.utils.tensorboard import SummaryWriter

# from cleanrl_utils.atari_wrappers import (
#     ClipRewardEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )

# MODEL_PATH = "/home/partha/birbal3d_project/cleanrl/cleanrl/runs/ALE/Breakout-v5__ppo_atari__1__1766512651/ppo_model.pt"
# EVAL_EPISODES = 9
# VIDEO_DIR = "videos/ppo_eval_final"
# ENV_ID = "ALE/Breakout-v5"


# def make_env(run_name=None):
#     env = gym.make(ENV_ID, render_mode="rgb_array")

#     env = gym.wrappers.RecordVideo(
#         env,
#         VIDEO_DIR,
#         episode_trigger=lambda i: True,
#         video_length=0,
#     )

#     env = gym.wrappers.RecordEpisodeStatistics(env)

#     env = NoopResetEnv(env, 30)
#     env = MaxAndSkipEnv(env, 4)

#     if "FIRE" in env.unwrapped.get_action_meanings():
#         env = FireResetEnv(env)

#     env = ClipRewardEnv(env)
#     env = gym.wrappers.ResizeObservation(env, (84, 84))
#     env = gym.wrappers.GrayScaleObservation(env)
#     env = gym.wrappers.FrameStack(env, 4)

#     return env


# # MUST MATCH CLEANRL PPO ARCHITECTURE
# class ActorCritic(torch.nn.Module):
#     def __init__(self, env):
#         super().__init__()

#         self.network = torch.nn.Sequential(
#             torch.nn.Conv2d(4, 32, 8, stride=4),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(32, 64, 4, stride=2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(64, 64, 3, stride=1),
#             torch.nn.ReLU(),
#             torch.nn.Flatten(),
#             torch.nn.Linear(3136, 512),
#             torch.nn.ReLU(),
#         )

#         self.actor = torch.nn.Linear(512, env.action_space.n)
#         self.critic = torch.nn.Linear(512, 1)

#     def act(self, x):
#         hidden = self.network(x / 255.0)
#         logits = self.actor(hidden)
#         probs = Categorical(logits=logits)
#         return probs.sample()


# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     run_name = "ppo_eval_final"
#     writer = SummaryWriter(f"runs/{run_name}")

#     env = make_env(run_name)
#     agent = ActorCritic(env).to(device)

#     print(f"Loading model: {MODEL_PATH}")
#     state = torch.load(MODEL_PATH, map_location=device)
#     agent.load_state_dict(state)

#     returns = []

#     for ep in range(EVAL_EPISODES):
#         obs, _ = env.reset()
#         obs = torch.tensor(obs, device=device)
#         done = False
#         total = 0

#         while not done:
#             with torch.no_grad():
#                 action = agent.act(obs.unsqueeze(0))

#             obs, r, term, trunc, _ = env.step(action.item())
#             obs = torch.tensor(obs, device=device)
#             total += r
#             done = term or trunc

#         print(f"Episode {ep}: return={total}")
#         writer.add_scalar("eval/episodic_return", total, ep)
#         returns.append(total)

#     env.close()

#     print("\n==== FINAL PPO EVALUATION ====")
#     print(f"mean: {np.mean(returns):.2f}")
#     print(f"min : {np.min(returns)}")
#     print(f"max : {np.max(returns)}")
#     print(f"std : {np.std(returns):.2f}")
# DQN Atari – Evaluation (9 episodes, same wrappers as training)

import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# ========= CHANGE ONLY THIS =========
MODEL_PATH = "/home/partha/birbal3d_project/cleanrl/cleanrl/runs/ALE/Breakout-v5__dqn_atari__1__1766257220/dqn_atari.cleanrl_model"
EVAL_EPISODES = 9
VIDEO = True
ENV_ID = "ALE/Breakout-v5"
# ====================================


def make_env(run_name=None):
    env = gym.make(ENV_ID, render_mode="rgb_array" if VIDEO else None)

    if VIDEO:
        env = gym.wrappers.RecordVideo(
            env,
            "videos/dqn_atari_eval_final",
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


class QNetwork(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, env.action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_name = "dqn_atari_eval_final"
    writer = SummaryWriter(f"runs/{run_name}")

    env = make_env(run_name)
    agent = QNetwork(env).to(device)

    print(f"Loading model: {MODEL_PATH}")
    state = torch.load(MODEL_PATH, map_location=device)
    agent.load_state_dict(state)

    returns = []

    for ep in range(EVAL_EPISODES):
        obs, _ = env.reset()
        obs = torch.tensor(obs, device=device)
        done = False
        total = 0

        while not done:
            with torch.no_grad():
                q = agent(obs.unsqueeze(0))
                action = torch.argmax(q, dim=1)

            obs, r, term, trunc, _ = env.step(action.item())
            obs = torch.tensor(obs, device=device)
            total += r
            done = term or trunc

        print(f"Episode {ep}: return={total}")
        writer.add_scalar("eval/episodic_return", total, ep)
        returns.append(total)

    env.close()

    print("\n==== FINAL DQN-ATARI EVALUATION ====")
    print(f"mean: {np.mean(returns):.2f}")
    print(f"min : {np.min(returns)}")
    print(f"max : {np.max(returns)}")
    print(f"std : {np.std(returns):.2f}")
