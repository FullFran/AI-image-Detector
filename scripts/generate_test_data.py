import os
import sys

sys.path.append(os.path.abspath("src"))
import numpy as np
from PIL import Image

from ai_detector.data.generation import (
    generate_natural_like_image,
    generate_synthetic_image,
)

os.makedirs("data/raw", exist_ok=True)

print("Generating test_real.png...")
img = generate_natural_like_image()
Image.fromarray((img * 255).astype(np.uint8)).save("data/raw/test_real.png")

print("Generating test_fake.png...")
img2 = generate_synthetic_image()
Image.fromarray((img2 * 255).astype(np.uint8)).save("data/raw/test_fake.png")

print("Done.")
