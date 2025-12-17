import numpy as np
from PIL import Image


def load_image_from_path(image_path: str) -> np.ndarray:
    """Carga una imagen desde un path y asegura formato RGB."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img)


def rgb_to_luminance(image: np.ndarray) -> np.ndarray:
    """Convierte imagen RGB a Luminancia (escala de grises) usando pesos estándar."""
    # Si ya es grayscale
    if len(image.shape) == 2:
        return (
            image.astype(np.float64) / 255.0
            if image.max() > 1
            else image.astype(np.float64)
        )

    if image.max() > 1:
        image = image.astype(np.float64) / 255.0

    luminance_weights = np.array([0.2126, 0.7152, 0.0722])
    return np.dot(image[..., :3], luminance_weights)
