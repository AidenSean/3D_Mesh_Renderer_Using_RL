import argparse
import sys
import os
import random
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from speech_module import SpeechProcessor
from genai_module import TextureGenerator
from voxel_module import ImageToVoxel
from rl_env import VoxelBuilderEnv
from napari_visualizer import run_visualization

def heuristic_agent(env):
    """
    A smart agent that knows exactly what to do.
    Used for the 'Live' demo to show building happening efficiently.
    """
    # Current pos
    x, y, z = env.agent_pos
    
    # 1. Is the current block correct?
    current_val = env.current_grid[x, y, z]
    target_val = env.target_grid[x, y, z]
    
    if current_val != target_val:
        if target_val == 1:
            return 6 # Place
        else:
            return 7 # Remove
            
    # 2. Find nearest incorrect block
    # Simple scan
    # Create list of discrepancies
    diff = env.target_grid - env.current_grid
    indices = np.argwhere(diff != 0)
    
    if len(indices) == 0:
        return 0 # No op / Dance
        
    # Find closest index
    current_pos_arr = np.array([x, y, z])
    distances = np.sum(np.abs(indices - current_pos_arr), axis=1)
    nearest_idx = np.argmin(distances)
    target_pos = indices[nearest_idx]
    
    # Move towards target
    tx, ty, tz = target_pos
    
    if tx > x: return 0 
    if tx < x: return 1
    if ty > y: return 2
    if ty < y: return 3
    if tz > z: return 4
    if tz < z: return 5
    
    return env.action_space.sample() # Should not reach here if logic is correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_prompt", type=str, default=None, help="Skip speech and use this prompt")
    args = parser.parse_args()
    
    # 1. Speech
    prompt = args.img_prompt
    if not prompt:
        print("Initializing Speech Recognition...")
        speech = SpeechProcessor()
        prompt = speech.listen_for_command()
        
    if not prompt:
        print("No prompt received. Exiting.")
        return

    print(f"Workflow starting for: {prompt}")

    # 2. Generation
    # Optimize: Check if image already exists?
    genai = TextureGenerator(device="mps") # Force MPS for M3 Pro
    image = genai.generate_pixel_art(prompt, size=(32, 32)) # 32x32 grid
    
    # 3. Voxelize
    print("Converting to Voxels...")
    converter = ImageToVoxel()
    voxels, colors = converter.convert(image)
    
    print(f"Voxel Grid Shape: {voxels.shape}")
    
    # Save Mesh
    mesh_path = converter.save_mesh(voxels, colors, filename=prompt.replace(" ", "_"))
    
    # 4. Environment
    print("Setting up RL Environment...")
    env = VoxelBuilderEnv(voxels, colors)
    obs, _ = env.reset()
    
    # 5. Visualize / Run
    print("Starting Visualization...")
    # Passing the heuristic agent to make it actually build something cool 'live'
    run_visualization(env, heuristic_agent, mesh_path=mesh_path)

if __name__ == "__main__":
    main()
