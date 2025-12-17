import numpy as np

from .color import ColorAnalyzer
from .dct import DCTAnalyzer
from .spectral import SpectralAnalyzer


class UnifiedAIImageDetector:
    """
    Detector unificado que combina múltiples análisis:
    - Covarianza de gradientes (implícita si se añade)
    - Análisis espectral (Fourier)
    - Análisis DCT
    - Análisis de color
    """

    def __init__(self):
        self.spectral_analyzer = SpectralAnalyzer()
        self.dct_analyzer = DCTAnalyzer()
        self.color_analyzer = ColorAnalyzer()

    def extract_all_features(self, image: np.ndarray) -> dict:
        """
        Extrae todas las features de una imagen.
        """
        features = {}

        # Análisis espectral
        spectral_features = self.spectral_analyzer.compute_spectral_features(image)
        features.update({f"spectral_{k}": v for k, v in spectral_features.items()})

        # Análisis DCT
        dct_result = self.dct_analyzer.analyze(image)
        features.update({f"dct_{k}": v for k, v in dct_result["features"].items()})

        # Análisis de color
        color_features = self.color_analyzer.compute_color_features(image)
        features.update({f"color_{k}": v for k, v in color_features.items()})

        return features
