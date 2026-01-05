from env.block_env import BlockEnv, GRID
from agent.dqn_agent import Agent
import torch

env = BlockEnv(render=False)
agent = Agent(GRID*GRID, GRID)

for episode in range(250):
    state = env.reset()
    total = 0

    while True:
        action = agent.act(state)
        state, reward, done = env.step(action)
        total += reward
        if done:
            break

    agent.epsilon *= 0.97
    print(f"Episode {episode} | Reward {total:.2f}")

torch.save(agent.model.state_dict(), "models/rl_builder.pth")
