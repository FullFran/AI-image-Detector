import numpy as np
from scipy import ndimage

from ..utils import rgb_to_luminance


class GradientCovarianceDetector:
    """
    Detector de imágenes generadas por IA basado en análisis de covarianza de gradientes.
    """

    def __init__(self):
        self.features_ = None
        self.covariance_matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None

    def compute_gradients(self, luminance: np.ndarray) -> tuple:
        Gx = ndimage.sobel(luminance, axis=1, mode="reflect")
        Gy = ndimage.sobel(luminance, axis=0, mode="reflect")
        return Gx, Gy

    def build_gradient_matrix(self, Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
        M = np.column_stack([Gx.flatten(), Gy.flatten()])
        return M

    def compute_covariance_matrix(self, M: np.ndarray) -> np.ndarray:
        N = M.shape[0]
        M_centered = M - M.mean(axis=0)
        C = (1 / N) * (M_centered.T @ M_centered)
        return C

    def extract_features(self, C: np.ndarray) -> dict:
        eigenvalues, eigenvectors = np.linalg.eigh(C)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

        lambda1, lambda2 = eigenvalues
        eps = 1e-10
        lambda2_safe = max(lambda2, eps)

        features = {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "trace": np.trace(C),
            "determinant": np.linalg.det(C),
            "eigenvalue_ratio": lambda1 / lambda2_safe,
            "anisotropy": (lambda1 - lambda2) / (lambda1 + lambda2 + eps),
            "eccentricity": np.sqrt(1 - lambda2_safe / lambda1) if lambda1 > eps else 0,
            "frobenius_norm": np.linalg.norm(C, "fro"),
            "condition_number": lambda1 / lambda2_safe,
            "covariance_xy": C[0, 1],
            "variance_x": C[0, 0],
            "variance_y": C[1, 1],
        }

        det_C = max(features["determinant"], eps)
        features["differential_entropy"] = 0.5 * np.log((2 * np.pi * np.e) ** 2 * det_C)

        return features

    def analyze(self, image: np.ndarray) -> dict:
        L = rgb_to_luminance(image)
        Gx, Gy = self.compute_gradients(L)
        M = self.build_gradient_matrix(Gx, Gy)
        C = self.compute_covariance_matrix(M)
        self.covariance_matrix_ = C
        features = self.extract_features(C)
        self.features_ = features

        return {
            "luminance": L,
            "gradient_x": Gx,
            "gradient_y": Gy,
            "gradient_matrix": M,
            "covariance_matrix": C,
            "features": features,
        }
