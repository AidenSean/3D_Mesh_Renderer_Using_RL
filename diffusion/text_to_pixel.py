import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_pixel_art(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)

    full_prompt = (
        f"pixel art of a {prompt}, 32x32, low resolution, "
        "blocky, simple colors, black background"
    )

    image = pipe(
        full_prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image = image.resize((32, 32), Image.NEAREST)
    image.save("assets/reference.png")
    print("Pixel art generated → assets/reference.png")

if __name__ == "__main__":
    obj = input("Enter object name: ")
    generate_pixel_art(obj)
