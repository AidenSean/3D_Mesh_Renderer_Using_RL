from env.block_env import BlockEnv, GRID
from agent.dqn_agent import Agent
import torch

env = BlockEnv(render=True)
agent = Agent(GRID*GRID, GRID)
agent.model.load_state_dict(torch.load("models/rl_builder.pth"))
agent.epsilon = 0

state = env.reset()

while True:
    state, _, done = env.step(agent.act(state))
    env.render()
    if done:
        break
