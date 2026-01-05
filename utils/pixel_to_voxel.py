from PIL import Image
import numpy as np

def pixel_to_voxel(path):
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr > 100).astype(int)
