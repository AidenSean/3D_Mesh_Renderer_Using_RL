import torch
from diffusers import DiffusionPipeline
from PIL import Image

class TextureGenerator:
    def __init__(self, device="mps"):
        self.device = device
        print(f"Loading GenAI Model on {self.device}...")
        
        # Using LCM (Latent Consistency Model) for extreme speed (4-8 steps)
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        
        try:
            self.pipe = DiffusionPipeline.from_pretrained(model_id)
            self.pipe.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please run src/download_model.py first.")
            raise e

    def generate_pixel_art(self, prompt, size=(64, 64)):
        """Generates a pixel art image based on the prompt."""
        
        enhanced_prompt = f"pixel art of a {prompt}, white background, isometric, 16-bit, simple, clean, centered, low resolution, minecraft style, blocky, 8k"
        
        print(f"Generating image for: {enhanced_prompt}")
        
        # LCM requires very few steps (4-8)
        image = self.pipe(
            prompt=enhanced_prompt, 
            num_inference_steps=6, # FAST!
            guidance_scale=8.0,
            height=512,
            width=512
        ).images[0]
        
        # Downscale to pixel art size
        pixel_image = image.resize(size, resample=Image.NEAREST)
        
        # Retrieve path
        save_path = f"assets/{prompt.replace(' ', '_')}.png"
        pixel_image.save(save_path)
        print(f"Image saved to {save_path}")
        
        return pixel_image
