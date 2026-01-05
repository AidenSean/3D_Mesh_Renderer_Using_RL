import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

device = "mps"

pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16
).to(device)

def generate_pixel_art(prompt):
    full_prompt = (
        f"pixel art of a {prompt}, 32x32, sprite, blocky, simple colors, black background"
    )

    image = pipe(
        full_prompt,
        num_inference_steps=1,
        guidance_scale=0.0
    ).images[0]

    image = image.resize((32, 32), Image.NEAREST)
    image.save("assets/reference.png")
    print("Pixel art generated → assets/reference.png")

if __name__ == "__main__":
    obj = input("Enter object name: ")
    generate_pixel_art(obj)
