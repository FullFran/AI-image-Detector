import numpy as np
from scipy.stats import entropy


class ColorAnalyzer:
    """
    Analizador de características de color para detección de IA.
    """

    def rgb_to_ycbcr(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convierte RGB a YCbCr (ITU-R BT.601).
        """
        if rgb.max() > 1:
            rgb = rgb.astype(np.float64) / 255.0

        transform = np.array(
            [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.500], [0.500, -0.419, -0.081]]
        )

        ycbcr = np.dot(rgb[..., :3], transform.T)
        ycbcr[..., 1:] += 0.5  # Offset para Cb, Cr

        return ycbcr

    def rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        """Convierte RGB a HSV."""
        if rgb.max() > 1:
            rgb = rgb.astype(np.float64) / 255.0

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        maxc = np.maximum(r, np.maximum(g, b))
        minc = np.minimum(r, np.minimum(g, b))

        v = maxc
        # Evitar division por cero cuando maxc=0
        s = np.where(maxc != 0, (maxc - minc) / maxc, 0)

        delta = maxc - minc
        # Evitar division por cero en delta
        delta_safe = np.where(delta == 0, 1, delta)

        h = np.zeros_like(r)

        mask_r = (maxc == r) & (maxc != minc)
        mask_g = (maxc == g) & (maxc != minc)
        mask_b = (maxc == b) & (maxc != minc)

        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta_safe[mask_r]) % 6
        h[mask_g] = (b[mask_g] - r[mask_g]) / delta_safe[mask_g] + 2
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta_safe[mask_b] + 4

        h = h / 6  # Normalizar a [0, 1]

        return np.stack([h, s, v], axis=-1)

    def compute_channel_correlations(self, image: np.ndarray) -> dict:
        """
        Calcula correlaciones entre canales de color.
        """
        if image.max() > 1:
            image = image.astype(np.float64) / 255.0

        r = image[..., 0].flatten()
        g = image[..., 1].flatten()
        b = image[..., 2].flatten()

        # Clip para evitar NaN si algun canal es constante
        return {
            "corr_rg": np.corrcoef(r, g)[0, 1] if r.std() > 0 and g.std() > 0 else 0,
            "corr_rb": np.corrcoef(r, b)[0, 1] if r.std() > 0 and b.std() > 0 else 0,
            "corr_gb": np.corrcoef(g, b)[0, 1] if g.std() > 0 and b.std() > 0 else 0,
        }

    def compute_color_features(self, image: np.ndarray) -> dict:
        """
        Extrae features de color completas.
        """
        if image.max() > 1:
            image = image.astype(np.float64) / 255.0

        # Espacios de color
        ycbcr = self.rgb_to_ycbcr(image)
        hsv = self.rgb_to_hsv(image)

        features = {}

        # Correlaciones RGB
        features.update(self.compute_channel_correlations(image))

        # Estadísticas YCbCr
        features["y_mean"] = ycbcr[..., 0].mean()
        features["y_std"] = ycbcr[..., 0].std()
        features["cb_std"] = ycbcr[..., 1].std()
        features["cr_std"] = ycbcr[..., 2].std()
        # Chroma energy
        features["chroma_energy"] = (ycbcr[..., 1] ** 2 + ycbcr[..., 2] ** 2).mean()

        # Estadísticas HSV
        features["saturation_mean"] = hsv[..., 1].mean()
        features["saturation_std"] = hsv[..., 1].std()
        features["value_mean"] = hsv[..., 2].mean()
        features["value_std"] = hsv[..., 2].std()

        # Entropía de histograma de hue
        h_hist, _ = np.histogram(hsv[..., 0].flatten(), bins=36, range=(0, 1))
        h_hist = h_hist.astype(float)
        h_sum = h_hist.sum()
        if h_sum > 0:
            h_hist = h_hist / h_sum
            features["hue_entropy"] = entropy(h_hist + 1e-10)
        else:
            features["hue_entropy"] = 0

        return features
