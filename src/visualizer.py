from ursina import *
import numpy as np

class VoxelVisualizer:
    def __init__(self, env, agent_algorithm=None):
        self.app = Ursina()
        
        self.env = env
        self.agent_algorithm = agent_algorithm
        
        # Camera setup
        self.cam = EditorCamera()
        self.cam.position = (env.grid_shape[0]//2, env.grid_shape[1]//2, -env.grid_shape[0]*2)
        
        # Grid visual
        self.blocks = {} # Key: (x,y,z), Value: Entity
        self.agent_entity = Entity(model='cube', color=color.red, scale=1.1, position=env.agent_pos)
        
        # Text
        self.info_text = Text(text="Say a word to start...", position=(-.45, .45), scale=2, origin=(0,0))
        
        # Reference ghost blocks
        self.show_target_ghosts()

        # Update Entity - Ursina calls 'update' on entities automatically
        self.update_entity = Entity()
        self.update_entity.update = self.step

    def show_target_ghosts(self):
        # Create semi-transparent blocks for target
        for x in range(self.env.grid_shape[0]):
            for y in range(self.env.grid_shape[1]):
                for z in range(self.env.grid_shape[2]):
                    if self.env.target_grid[x, y, z] == 1:
                        c = self.env.target_colors[x, y, z]
                        col = color.rgba(c[0], c[1], c[2], 50) # Transparent
                        Entity(model='cube', color=col, position=(x,y,z), scale=0.9, alpha=0.2)
    
    def step(self):
         # 1. Get Action from Agent
        if self.agent_algorithm:
            action = self.agent_algorithm(self.env)
        else:
            action = self.env.action_space.sample() # Random agent
            
        # 2. Step Environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Update Visuals
        # Update Agent Position
        self.agent_entity.position = self.env.agent_pos
        self.agent_entity.animate_position(self.env.agent_pos, duration=0.1)
        
        # Update Blocks
        ax, ay, az = int(self.env.agent_pos[0]), int(self.env.agent_pos[1]), int(self.env.agent_pos[2])
        block_key = (ax, ay, az)
        
        if self.env.current_grid[ax, ay, az] == 1:
            if block_key not in self.blocks:
                c = self.env.current_colors[ax, ay, az]
                col = color.rgb(c[0], c[1], c[2])
                self.blocks[block_key] = Entity(model='cube', color=col, position=(ax, ay, az))
        else:
            if block_key in self.blocks:
                destroy(self.blocks[block_key])
                del self.blocks[block_key]
                
        if terminated:
            self.info_text.text = "Build Complete!"
            # Stop updating to save resources or just return
            # self.update_entity.ignore = True 

    def run(self):
        self.app.run()

def run_visualization(env, agent_func):
    viz = VoxelVisualizer(env, agent_func)
    viz.run()
