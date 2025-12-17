import numpy as np
from scipy import ndimage
from scipy.stats import entropy

from ..utils import rgb_to_luminance


class SpectralAnalyzer:
    """
    Analizador espectral para detección de imágenes generadas por IA.

    Implementa análisis en dominio de Fourier, extracción de estadísticas
    espectrales, y detección de artefactos periódicos.
    """

    def __init__(self):
        self.spectrum_ = None
        self.power_spectrum_ = None
        self.azimuthal_average_ = None

    def compute_fft(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula la FFT 2D centrada de la imagen.
        """
        gray = rgb_to_luminance(image)

        # FFT 2D
        fft = np.fft.fft2(gray)

        # Centrar (mover DC al centro)
        fft_shifted = np.fft.fftshift(fft)

        return fft_shifted

    def compute_power_spectrum(self, fft: np.ndarray) -> np.ndarray:
        """
        Calcula el espectro de potencia.
        P(u,v) = |F(u,v)|²
        """
        return np.abs(fft) ** 2

    def azimuthal_average(self, spectrum: np.ndarray) -> tuple:
        """
        Calcula el promedio azimutal del espectro (perfil radial).
        """
        H, W = spectrum.shape
        cy, cx = H // 2, W // 2

        # Crear mapa de distancias radiales
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

        # Calcular promedio para cada radio
        r_max = min(cx, cy)
        # Optimizacion vectorizada usando bincount
        r_flat = r.flatten()
        spec_flat = spectrum.flatten()

        # Solo considerar radios validos
        valid_mask = r_flat < r_max
        r_valid = r_flat[valid_mask]
        spec_valid = spec_flat[valid_mask]

        radial_sum = np.bincount(r_valid, weights=spec_valid)
        counts = np.bincount(r_valid)

        # Evitar division por cero
        radial_profile = np.zeros_like(radial_sum, dtype=np.float64)
        np.divide(radial_sum, counts, out=radial_profile, where=counts > 0)

        frequencies = np.arange(len(radial_profile))

        return frequencies, radial_profile

    def fit_power_law(self, frequencies: np.ndarray, profile: np.ndarray) -> dict:
        """
        Ajusta ley de potencias P(f) = A / f^α al perfil radial.
        """
        # Evitar f=0 y valores muy pequeños
        valid = (frequencies > 1) & (profile > 0)
        f_valid = frequencies[valid]
        p_valid = profile[valid]

        if len(f_valid) < 10:
            return {"alpha": np.nan, "A": np.nan, "r_squared": 0}

        # Regresión lineal en log-log
        log_f = np.log(f_valid)
        log_p = np.log(p_valid)

        # Ajuste por mínimos cuadrados
        coeffs = np.polyfit(log_f, log_p, 1)
        alpha = -coeffs[0]  # Pendiente negativa
        log_A = coeffs[1]
        A = np.exp(log_A)

        # R² del ajuste
        log_p_pred = coeffs[0] * log_f + coeffs[1]
        ss_res = np.sum((log_p - log_p_pred) ** 2)
        ss_tot = np.sum((log_p - log_p.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {"alpha": alpha, "A": A, "r_squared": r_squared}

    def detect_periodic_artifacts(
        self, spectrum: np.ndarray, threshold_sigma: float = 5.0
    ) -> dict:
        """
        Detecta picos periódicos anómalos en el espectro.
        """
        log_spectrum = np.log(spectrum + 1e-10)

        # Suavizar para obtener baseline
        baseline = ndimage.gaussian_filter(log_spectrum, sigma=5)

        # Residuos
        residuals = log_spectrum - baseline

        # Detectar outliers
        mean_res = residuals.mean()
        std_res = residuals.std()

        peaks_mask = residuals > (mean_res + threshold_sigma * std_res)
        n_peaks = peaks_mask.sum()

        # Energía en picos vs total
        peak_energy_ratio = (
            spectrum[peaks_mask].sum() / spectrum.sum() if n_peaks > 0 else 0
        )

        return {
            "n_peaks": n_peaks,
            "peak_energy_ratio": peak_energy_ratio,
            "max_residual": residuals.max(),
            "peaks_mask": peaks_mask,
        }

    def compute_spectral_features(self, image: np.ndarray) -> dict:
        """
        Extrae todas las features espectrales de una imagen.
        """
        # FFT y espectro de potencia
        fft = self.compute_fft(image)
        power = self.compute_power_spectrum(fft)

        self.spectrum_ = np.abs(fft)
        self.power_spectrum_ = power

        # Perfil radial
        freqs, profile = self.azimuthal_average(power)
        self.azimuthal_average_ = (freqs, profile)

        # Ajuste ley de potencias
        power_law = self.fit_power_law(freqs, profile)

        # Detección de artefactos
        artifacts = self.detect_periodic_artifacts(power)

        # Estadísticas adicionales
        H, W = power.shape
        cy, cx = H // 2, W // 2

        # Energía en diferentes bandas de frecuencia
        y, x = np.ogrid[:H, :W]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_max = min(cx, cy)

        low_freq_mask = r < r_max * 0.1
        mid_freq_mask = (r >= r_max * 0.1) & (r < r_max * 0.5)
        high_freq_mask = r >= r_max * 0.5

        total_energy = power.sum()

        # Helpers seguros
        def safe_sum_ratio(mask):
            return power[mask].sum() / total_energy if total_energy > 0 else 0

        features = {
            # Ley de potencias
            "spectral_alpha": power_law["alpha"],
            "power_law_r2": power_law["r_squared"],
            # Distribución de energía
            "low_freq_energy_ratio": safe_sum_ratio(low_freq_mask),
            "mid_freq_energy_ratio": safe_sum_ratio(mid_freq_mask),
            "high_freq_energy_ratio": safe_sum_ratio(high_freq_mask),
            # Artefactos periódicos
            "n_spectral_peaks": artifacts["n_peaks"],
            "peak_energy_ratio": artifacts["peak_energy_ratio"],
            # Estadísticas globales
            "spectral_entropy": entropy(power.flatten() / power.sum())
            if power.sum() > 0
            else 0,
        }

        # Centroide y spread
        if profile.sum() > 0:
            centroid = (freqs * profile).sum() / profile.sum()
            spread = np.sqrt(((freqs - centroid) ** 2 * profile).sum() / profile.sum())
            features["spectral_centroid"] = centroid
            features["spectral_spread"] = spread
        else:
            features["spectral_centroid"] = 0
            features["spectral_spread"] = 0

        return features
