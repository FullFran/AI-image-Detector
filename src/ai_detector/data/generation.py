import numpy as np
import pandas as pd
from scipy import ndimage

from ..features.gradient import GradientCovarianceDetector


def generate_natural_like_image(size=(256, 256), seed=42):
    """Genera imagen con estadísticas 1/f naturales."""
    np.random.seed(seed)
    H, W = size

    u = np.fft.fftfreq(H)[:, np.newaxis]
    v = np.fft.fftfreq(W)[np.newaxis, :]
    freq = np.sqrt(u**2 + v**2)
    freq[0, 0] = 1

    alpha = 2.0
    spectrum = 1 / (freq**alpha)
    spectrum[0, 0] = 0

    phase = np.random.uniform(0, 2 * np.pi, (H, W))
    fourier = spectrum * np.exp(1j * phase)
    img_gray = np.real(np.fft.ifft2(fourier))
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())

    img = np.stack(
        [img_gray * 0.8 + 0.1, img_gray * 0.9 + 0.05, img_gray * 0.7 + 0.15], axis=-1
    )

    return np.clip(img, 0, 1)


def generate_synthetic_image(size=(256, 256), seed=42):
    """Simula imagen generada por IA con artefactos típicos."""
    np.random.seed(seed)
    H, W = size

    base = np.random.randn(H // 4, W // 4)
    base = ndimage.zoom(base, 4, order=3)

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    grid_pattern = 0.02 * np.sin(x * np.pi / 8) * np.sin(y * np.pi / 8)

    img_gray = base + grid_pattern
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    img_gray = ndimage.gaussian_filter(img_gray, sigma=1.5)

    img = np.stack([img_gray, img_gray, img_gray], axis=-1)

    return np.clip(img, 0, 1)


def generate_training_dataset(n_samples=100):
    """Genera dataset de entrenamiento sintético."""
    detector = GradientCovarianceDetector()
    data = []

    for i in range(n_samples):
        # Imagen natural
        img = generate_natural_like_image(size=(128, 128), seed=i)
        result = detector.analyze(img)
        features = result["features"].copy()
        features["label"] = 0  # 0 = natural
        data.append(features)

        # Imagen sintética
        img = generate_synthetic_image(size=(128, 128), seed=i + 1000)
        result = detector.analyze(img)
        features = result["features"].copy()
        features["label"] = 1  # 1 = sintética
        data.append(features)

    return pd.DataFrame(data)
