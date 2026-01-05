import gymnasium as gym
import torch
from cleanrl.dqn_atari import QNetwork

env = gym.make(
    "ALE/Breakout-v5",
    render_mode="rgb_array"
)

env = gym.wrappers.RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda episode_id: True
)

obs, _ = env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

q_network = QNetwork(env).to(device)
q_network.eval()

done = False
while not done:
    obs_t = torch.tensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        action = q_network(obs_t).argmax(dim=1).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

env.close()
