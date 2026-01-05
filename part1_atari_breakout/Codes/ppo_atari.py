# PPO Atari — Train + Save + Evaluate (Eval Only Supported)

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
    exp_name: str = "ppo_atari"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    capture_video: bool = False

    env_id: str = "ALE/Breakout-v5"
    total_timesteps: int = 5_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None

    save_model: bool = True
    eval_episodes: int = 10

    # NEW — evaluation controls
    eval_only: bool = False
    model_path: str = ""


# =========================
# Environment
# =========================
def make_env(env_id, idx, capture_video, run_name):
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
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


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
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ---- load pretrained if eval only ----
    if args.eval_only:
        assert args.model_path != "", "You must pass --model-path when eval_only=True"
        state = torch.load(args.model_path, map_location=device)
        agent.load_state_dict(state)
        print(f"Loaded PPO model from: {args.model_path}")

    obs_shape = envs.single_observation_space.shape
    global_step = 0

    # =========================
    # TRAINING
    # =========================
    if not args.eval_only:
        obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs, dtype=torch.float32).to(device)

        num_iterations = args.total_timesteps // (args.num_envs * args.num_steps)

        for iteration in range(num_iterations):
            # rollout
            for step in range(args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)

                actions[step] = action
                logprobs[step] = logprob
                values[step] = value.flatten()

                next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
                next_done = torch.tensor(
                    np.logical_or(terminated, truncated),
                    dtype=torch.float32,
                    device=device,
                )
                rewards[step] = torch.tensor(reward).to(device)
                next_obs = torch.tensor(next_obs).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(
                                f"global_step={global_step}, "
                                f"episodic_return={info['episode']['r']}"
                            )
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

            # compute returns
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = (
                        delta
                        + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + values

            # flatten
            b_obs = obs.reshape((-1,) + obs_shape)
            b_actions = actions.reshape(-1)
            b_logprobs = logprobs.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            batch_size = b_obs.shape[0]
            minibatch_size = batch_size // args.num_minibatches
            inds = np.arange(batch_size)

            for _ in range(args.update_epochs):
                np.random.shuffle(inds)
                for start in range(0, batch_size, minibatch_size):
                    mb_inds = inds[start:start + minibatch_size]
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_inds],
                        b_actions.long()[mb_inds],
                    )

                    ratio = (newlogprob - b_logprobs[mb_inds]).exp()

                    adv = b_advantages[mb_inds]
                    if args.norm_adv:
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                    pg_loss = -torch.min(
                        adv * ratio,
                        adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef),
                    ).mean()

                    v_loss = 0.5 * (b_returns[mb_inds] - newvalue.view(-1)).pow(2).mean()
                    entropy_loss = entropy.mean()

                    loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

        # save model
        if args.save_model:
            model_path = f"runs/{run_name}/ppo_model.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"Saved PPO model to: {model_path}")

    # =========================
    # EVALUATION
    # =========================
    # def make_eval_env():
    #     env = gym.make(args.env_id, render_mode="rgb_array")
    #     env = gym.wrappers.RecordVideo(
    #         env,
    #         f"videos/{run_name}-eval",
    #         episode_trigger=lambda x: True,
    #     )
    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     env = NoopResetEnv(env, 30)
    #     env = MaxAndSkipEnv(env, 4)
    #     env = EpisodicLifeEnv(env)
    #     if "FIRE" in env.unwrapped.get_action_meanings():
    #         env = FireResetEnv(env)
    #     env = ClipRewardEnv(env)
    #     env = gym.wrappers.ResizeObservation(env, (84, 84))
    #     env = gym.wrappers.GrayScaleObservation(env)
    #     env = gym.wrappers.FrameStack(env, 4)
    #     return env
    # def make_eval_env():
    #     env = gym.make(args.env_id, render_mode="rgb_array")

    #     env = gym.wrappers.RecordVideo(
    #         env,
    #         f"videos/{run_name}-eval",
    #         episode_trigger=lambda episode_idx: episode_idx < args.eval_episodes,
    #         video_length=0,
    #     )

    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     env = NoopResetEnv(env, 30)
    #     env = MaxAndSkipEnv(env, 4)
    #     env = EpisodicLifeEnv(env)
    #     if "FIRE" in env.unwrapped.get_action_meanings():
    #         env = FireResetEnv(env)
    #     env = ClipRewardEnv(env)
    #     env = gym.wrappers.ResizeObservation(env, (84, 84))
    #     env = gym.wrappers.GrayScaleObservation(env)
    #     env = gym.wrappers.FrameStack(env, 4)
    #     return env
    def make_eval_env():
        env = gym.make(args.env_id, render_mode="rgb_array")

        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}-eval",
            episode_trigger=lambda x: True
        )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, 30)
        env = MaxAndSkipEnv(env, 4)

        # 🚨 REMOVE EpisodicLifeEnv for evaluation!
        # env = EpisodicLifeEnv(env)

        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)

        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        return env


    eval_env = make_eval_env()

    for ep in range(args.eval_episodes):
        obs, _ = eval_env.reset()
        obs = torch.tensor(obs).to(device)
        done = False
        total_reward = 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(obs.unsqueeze(0))
            obs, r, term, trunc, _ = eval_env.step(action.item())
            obs = torch.tensor(obs).to(device)
            done = term or trunc
            total_reward += r

        writer.add_scalar("eval/episodic_return", total_reward, ep)

    eval_env.close()
    envs.close()
    writer.close()
