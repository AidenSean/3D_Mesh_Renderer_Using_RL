import napari
import numpy as np
from PyQt5.QtCore import QTimer
import trimesh

class NapariVisualizer:
    def __init__(self, env, agent_algorithm=None, mesh_path=None):
        self.env = env
        self.agent_algorithm = agent_algorithm
        
        print("Initializing Napari Viewer...")
        self.viewer = napari.Viewer(title="Pixel RL 3D Builder")
        
        # 1. Target (Ghost) - REMOVED as per user request
        
        # 2. Current Build (Empty initially)
        self.build_layer = self.viewer.add_points(
            np.empty((0, 3)),
            face_color='white',
            size=0.9,
            name="Current Build",
            symbol="square"
        )
        
        # 3. Agent
        self.agent_layer = self.viewer.add_points(
            np.array([self.env.agent_pos]),
            face_color='red',
            size=1.1,
            name="Builder Agent",
            symbol="diamond"
        )
        
        self.mesh_path = mesh_path
        
        # Timer for loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(50) # 50ms = 20fps

    def show_mesh(self):
        if self.mesh_path:
            try:
                print(f"Loading Mesh for display: {self.mesh_path}")
                mesh = trimesh.load(self.mesh_path)
                self.mesh_viewer = napari.Viewer(title="Smoothed 3D Model")
                
                # Napari add_surface takes (vertices, faces, values)
                # We can just provide (vertices, faces)
                self.mesh_viewer.add_surface(
                    (mesh.vertices, mesh.faces),
                    name="Smoothed Mesh",
                    colormap='gray',
                    opacity=1.0
                )
            except Exception as e:
                print(f"Could not load mesh for viewing: {e}")

    def update_loop(self):
        # 1. Agent Agent
        if self.agent_algorithm:
            try:
                action = self.agent_algorithm(self.env)
            except Exception:
                # Fallback if heuristic fails or bounds error
                action = self.env.action_space.sample()
        else:
            action = self.env.action_space.sample()
            
        # 2. Step
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. Update Visuals
        
        # Agent
        self.agent_layer.data = np.array([self.env.agent_pos])
        
        # Build 
        if action >= 6:
            current_indices = np.argwhere(self.env.current_grid == 1)
            if len(current_indices) > 0:
                current_colors = self.env.current_colors[current_indices[:, 0], current_indices[:, 1], current_indices[:, 2]] / 255.0
                self.build_layer.data = current_indices
                self.build_layer.face_color = current_colors
            else:
                self.build_layer.data = np.empty((0, 3))
        
        if terminated:
            print("Building Complete!")
            self.timer.stop()
            # Show smoothed mesh ONLY after building is complete
            self.show_mesh()

    def run(self):
        napari.run()

def run_visualization(env, agent_func, mesh_path=None):
    viz = NapariVisualizer(env, agent_func, mesh_path)
    viz.run()
