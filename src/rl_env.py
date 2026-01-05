import gymnasium as gym
from gymnasium import spaces
import numpy as np

class VoxelBuilderEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, target_voxel_grid, target_color_grid):
        super(VoxelBuilderEnv, self).__init__()
        
        self.target_grid = target_voxel_grid # Shape (W, H, D)
        self.target_colors = target_color_grid # Shape (W, H, D, 3)
        self.grid_shape = self.target_grid.shape
        
        self.current_grid = np.zeros(self.grid_shape, dtype=int)
        self.current_colors = np.zeros((*self.grid_shape, 3), dtype=int)
        
        # Agent position (x, y, z)
        self.agent_pos = [0, 0, 0]
        
        # Action space: 
        # 0: Move X+
        # 1: Move X-
        # 2: Move Y+
        # 3: Move Y-
        # 4: Move Z+
        # 5: Move Z-
        # 6: Place Block (takes color from target at this location)
        # 7: Remove Block
        self.action_space = spaces.Discrete(8)
        
        # Observation space: 
        # Returns [x, y, z, current_val, target_val]
        self.observation_space = spaces.Box(low=0, high=255, shape=(5,), dtype=np.float32)

        self.max_steps = self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2] * 2
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_grid = np.zeros(self.grid_shape, dtype=int)
        self.current_colors = np.zeros((*self.grid_shape, 3), dtype=int)
        self.agent_pos = [0, 0, 0]
        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # returns [x, y, z, current_at_pos, target_at_pos]
        x, y, z = self.agent_pos
        # Boundary check for observation
        if 0 <= x < self.grid_shape[0] and 0 <= y < self.grid_shape[1] and 0 <= z < self.grid_shape[2]:
            current_val = self.current_grid[x, y, z]
            target_val = self.target_grid[x, y, z]
        else:
            current_val = -1
            target_val = -1
            
        return np.array([x, y, z, current_val, target_val], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        reward = -0.01 # Step cost
        terminated = False
        truncated = False
        
        x, y, z = self.agent_pos
        
        # Movement
        if action == 0: x += 1
        elif action == 1: x -= 1
        elif action == 2: y += 1
        elif action == 3: y -= 1
        elif action == 4: z += 1
        elif action == 5: z -= 1
        
        # Clamp position
        x = max(0, min(self.grid_shape[0]-1, x))
        y = max(0, min(self.grid_shape[1]-1, y))
        z = max(0, min(self.grid_shape[2]-1, z))
        
        self.agent_pos = [x, y, z]
        
        # Building
        if action == 6: # Place Block
            if self.target_grid[x, y, z] == 1 and self.current_grid[x, y, z] == 0:
                self.current_grid[x, y, z] = 1
                self.current_colors[x, y, z] = self.target_colors[x, y, z]
                reward += 1.0 # Good job
            elif self.current_grid[x, y, z] == 1:
                reward -= 0.1 # Already there
            elif self.target_grid[x, y, z] == 0:
                self.current_grid[x, y, z] = 1 # Wrong placement
                reward -= 0.5 
                
        elif action == 7: # Remove Block
            if self.current_grid[x, y, z] == 1:
                self.current_grid[x, y, z] = 0
                if self.target_grid[x, y, z] == 0:
                    reward += 0.5 # Fixed a mistake
                else:
                    reward -= 1.0 # Removed a needed block
        
        # Check completion
        if np.array_equal(self.current_grid, self.target_grid):
            reward += 10
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        pass
