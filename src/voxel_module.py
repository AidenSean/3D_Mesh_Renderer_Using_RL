import numpy as np
from PIL import Image
import torch
from transformers import pipeline
import trimesh
from skimage import measure
import os

class ImageToVoxel:
    def __init__(self, device="mps"):
        self.device = device
        print("Loading Depth Estimation Model...")
        # using a small but good depth model
        self.depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf", device=device)

    def convert(self, image: Image.Image, voxel_size=(32, 32, 32)):
        """
        Converts a PIL image to a 3D voxel grid using Depth Estimation.
        """
        # 1. Get Color Data (Texture)
        # Resize image to X,Y of voxel grid
        w, h, d = voxel_size
        image_resized = image.resize((w, h), resample=Image.NEAREST).convert("RGB")
        color_data = np.array(image_resized) # (H, W, 3)
        
        # 2. Estimate Depth
        print("Estimating 3D structure...")
        depth_output = self.depth_estimator(image)
        depth_map = depth_output["depth"]
        # Resize depth to match voxel grid
        depth_map = depth_map.resize((w, h), resample=Image.NEAREST)
        depth_array = np.array(depth_map) # (H, W)
        
        # Normalize depth to 0..D range
        # Depth is usually inverse (lighter is closer). 
        # Check model: Depth Anything: lighter is closer? Usually yes.
        # Let's normalize it to [0, d-1]
        
        d_min = depth_array.min()
        d_max = depth_array.max()
        if d_max - d_min == 0:
            depth_normalized = np.zeros_like(depth_array) + d//2
        else:
            depth_normalized = (depth_array - d_min) / (d_max - d_min)
            depth_normalized = depth_normalized * (d - 1)
            
        depth_normalized = depth_normalized.astype(int)
        
        # 3. Construct 3D Grid
        voxel_grid = np.zeros((w, h, d), dtype=int)
        color_grid = np.zeros((w, h, d, 3), dtype=int)
        
        # We will create a "shell" or a "volume"
        # Let's create a volume where we fill from back to the depth value? 
        # Or just the surface?
        # User wants "blocks" like Minecraft. Usually that's solid.
        # Let's simple extrude: Place a block at (x, y, z_depth).
        # Maybe add thickness?
        
        print("Building Voxel Grid...")
        thickness = 3 # Add some thickness to the surface
        
        for y in range(h):
            for x in range(w):
                # Check original image for transparency (if available) - we forced RGB above so check brightness?
                # Usually generated pixel art has white background.
                r, g, b = color_data[y, x]
                # Simple white background removal
                if r > 240 and g > 240 and b > 240:
                    continue
                
                # Invert Y for 3D coords usually
                target_y = h - 1 - y
                target_z = depth_normalized[y, x]
                
                # Fill
                for t in range(thickness):
                    z = target_z - t
                    if 0 <= z < d:
                        voxel_grid[x, target_y, z] = 1
                        color_grid[x, target_y, z] = [r, g, b]
                        
        count = np.sum(voxel_grid)
        print(f"Generated {count} voxels.")
        return voxel_grid, color_grid

    def save_mesh(self, voxel_grid, color_grid, filename="output_mesh"):
        """
        Smoothens the voxel grid into a 3D mesh and saves it.
        """
        print("Smoothening to 3D Mesh...")
        
        # Use Marching Cubes to generate mesh
        # Pad grid to ensure closed mesh
        padded_grid = np.pad(voxel_grid, 1, mode='constant', constant_values=0)
        
        try:
            verts, faces, normals, values = measure.marching_cubes(padded_grid, level=0.5)
            
            # Create Trimesh object
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
            
            # Apply simple smoothing
            trimesh.smoothing.filter_laplacian(mesh, iterations=5)
            
            # Simple coloring (avg color of voxel?)
            # For now, let's export just the shape as coloring vertices from voxels 
            # after marching cubes is non-trivial without interpolation.
            # We can create a colored mesh but standard PLY/OBJ support for vertex colors varies.
            
            # Save
            os.makedirs("assets/3d_objects", exist_ok=True)
            save_path = f"assets/3d_objects/{filename}.obj"
            mesh.export(save_path)
            print(f"3D Mesh saved to {save_path}")
            
            # Also save as GLB
            save_path_glb = f"assets/3d_objects/{filename}.glb"
            mesh.export(save_path_glb)
            print(f"3D Mesh saved to {save_path_glb}")
            
            return save_path
            
        except Exception as e:
            print(f"Failed to generate mesh: {e}")
            return None
