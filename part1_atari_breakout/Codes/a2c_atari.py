# docs reference: A2C Atari (custom, CleanRL-style)

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# =========================
# Args
# =========================
@dataclass
class Args:
    exp_name: str = "a2c_atari"
    seed: int = 1
    cuda: bool = True
    capture_video: bool = False

    env_id: str = "ALE/Breakout-v5"
    total_timesteps: int = 5_000_000

    num_envs: int = 8
    num_steps: int = 5
    learning_rate: float = 7e-4
    gamma: float = 0.99
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # NEW
    save_model: bool = True
    eval_episodes: int = 10


# =========================
# Environment
# =========================
def make_env(env_id, idx, run_name, capture_video):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, 30)
        env = MaxAndSkipEnv(env, 4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


# =========================
# Model
# =========================
class ActorCritic(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
        )
        self.policy = nn.Linear(512, envs.single_action_space.n)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        h = self.net(x)
        return self.policy(h), self.value(h)


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = tyro.cli(Args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, run_name, args.capture_video) for i in range(args.num_envs)]
    )

    model = ActorCritic(envs).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.tensor(obs, device=device)

    global_step = 0
    start_time = time.time()

    while global_step < args.total_timesteps:
        log_probs = []
        values = []
        rewards = []
        dones = []
        entropies = []

        for _ in range(args.num_steps):
            global_step += args.num_envs

            logits, value = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            entropies.append(dist.entropy())
            values.append(value.squeeze(-1))

            obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            rewards.append(torch.tensor(reward, device=device))
            dones.append(torch.tensor(done, dtype=torch.float32, device=device))

            obs = torch.tensor(obs, device=device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"step={global_step}, return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        with torch.no_grad():
            _, next_value = model(obs)
            next_value = next_value.squeeze(-1)

        returns = []
        R = next_value
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + args.gamma * R * (1.0 - d)
            returns.insert(0, R)

        returns = torch.stack(returns)
        values = torch.stack(values)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)

        advantages = returns - values

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        entropy_loss = entropies.mean()

        loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if global_step % 1000 == 0:
            sps = int(global_step / (time.time() - start_time))
            print("SPS:", sps)
            writer.add_scalar("charts/SPS", sps, global_step)

    # =========================
    # SAVE + EVALUATE
    # =========================
    if args.save_model:
        model_path = f"runs/{run_name}/a2c_model.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved A2C model to {model_path}")

        eval_env = make_env(
            args.env_id,
            idx=0,
            run_name=f"{run_name}-eval",
            capture_video=True,
        )()

        for ep in range(args.eval_episodes):
            obs, _ = eval_env.reset()
            obs = torch.tensor(obs, device=device)
            done = False
            total_reward = 0

            while not done:
                with torch.no_grad():
                    logits, _ = model(obs.unsqueeze(0))
                    action = torch.argmax(logits, dim=1)

                obs, r, term, trunc, _ = eval_env.step(action.item())
                obs = torch.tensor(obs, device=device)
                done = term or trunc
                total_reward += r

            writer.add_scalar("eval/episodic_return", total_reward, ep)

        eval_env.close()

    envs.close()
    writer.close()