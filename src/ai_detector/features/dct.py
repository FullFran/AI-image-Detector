import numpy as np
from scipy import fftpack

from ..utils import rgb_to_luminance


class DCTAnalyzer:
    """
    Analizador basado en DCT para detección de imágenes IA.

    Implementa análisis de coeficientes DCT en bloques estilo JPEG,
    extracción de histogramas de coeficientes, y detección de
    anomalías estadísticas.
    """

    BLOCK_SIZE = 8

    # Matrices de cuantización JPEG estándar (calidad 50)
    JPEG_QUANT_MATRIX = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )

    def __init__(self):
        self.dct_blocks_ = None
        self.coefficient_histograms_ = None

    def dct2(self, block: np.ndarray) -> np.ndarray:
        """
        DCT-II 2D de un bloque.
        DCT_2D = DCT_1D(filas) @ DCT_1D(columnas)
        """
        return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")

    def extract_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Extrae bloques 8x8 de la imagen.
        """
        gray = rgb_to_luminance(image)
        H, W = gray.shape

        # Truncar a múltiplos de 8
        H = (H // self.BLOCK_SIZE) * self.BLOCK_SIZE
        W = (W // self.BLOCK_SIZE) * self.BLOCK_SIZE
        gray = gray[:H, :W]

        # Reshape en bloques
        n_blocks_y = H // self.BLOCK_SIZE
        n_blocks_x = W // self.BLOCK_SIZE

        blocks = gray.reshape(n_blocks_y, self.BLOCK_SIZE, n_blocks_x, self.BLOCK_SIZE)
        blocks = blocks.transpose(0, 2, 1, 3)  # (ny, nx, 8, 8)

        return blocks

    def compute_dct_blocks(self, image: np.ndarray) -> np.ndarray:
        """
        Calcula DCT de todos los bloques 8x8.
        """
        blocks = self.extract_blocks(image)
        ny, nx, _, _ = blocks.shape

        dct_blocks = np.zeros_like(blocks)

        for i in range(ny):
            for j in range(nx):
                # Centrar bloque (como JPEG)
                block = blocks[i, j] - 0.5
                dct_blocks[i, j] = self.dct2(block)

        self.dct_blocks_ = dct_blocks
        return dct_blocks

    def extract_coefficient_statistics(self, dct_blocks: np.ndarray) -> dict:
        """
        Extrae estadísticas de los coeficientes DCT.
        """
        ny, nx, _, _ = dct_blocks.shape
        n_blocks = ny * nx

        # Aplanar bloques
        flat_blocks = dct_blocks.reshape(n_blocks, 8, 8)

        # Estadísticas agregadas
        dc_coeffs = flat_blocks[:, 0, 0]
        # Todos excepto DC
        ac_coeffs = flat_blocks[:, 0, 0] * 0  # Init con zeros para size
        ac_coeffs = flat_blocks.copy()
        ac_coeffs[:, 0, 0] = 0
        ac_coeffs = ac_coeffs.flatten()
        # ac_coeffs contains strict zeros at DC positions? No, flat_blocks copy
        # Let's do properly
        mask = np.ones((8, 8), dtype=bool)
        mask[0, 0] = False
        ac_coeffs = flat_blocks[:, mask].flatten()

        # Energía AC total per block
        # sum of squares of all coeffs minus DC squared
        ac_energy = (flat_blocks**2).sum(axis=(1, 2)) - flat_blocks[:, 0, 0] ** 2

        features = {
            "dc_mean": dc_coeffs.mean(),
            "dc_std": dc_coeffs.std(),
            "ac_energy_mean": ac_energy.mean(),
            "ac_energy_std": ac_energy.std(),
            "ac_mean": np.abs(ac_coeffs).mean(),
            "ac_std": ac_coeffs.std(),
            "ac_kurtosis": self._kurtosis(ac_coeffs),
            "high_freq_ratio": self._high_freq_ratio(flat_blocks),
            "blockiness": self._compute_blockiness(dct_blocks),
        }

        return features

    def _kurtosis(self, x: np.ndarray) -> float:
        """Calcula kurtosis (Fisher)."""
        n = len(x)
        if n < 4:
            return 0
        m = x.mean()
        s = x.std()
        if s == 0:
            return 0
        return ((x - m) ** 4).mean() / s**4 - 3

    def _high_freq_ratio(self, blocks: np.ndarray) -> float:
        """
        Ratio de energía en coeficientes de alta frecuencia (u + v >= 8).
        """
        total_energy = (blocks**2).sum()

        high_freq_energy = 0
        # Optimizar si es posible, por ahora bucle simple es claro
        # Hacerlo vectorizado usando indices
        u, v = np.meshgrid(np.arange(8), np.arange(8), indexing="ij")
        mask = (u + v) >= 8
        # blocks shape: (N, 8, 8)
        high_freq_energy = (blocks[:, mask] ** 2).sum()

        return high_freq_energy / total_energy if total_energy > 0 else 0

    def _compute_blockiness(self, dct_blocks: np.ndarray) -> float:
        """
        Mide el "blockiness" (artefactos de bloques) de la imagen.
        """
        ny, nx, _, _ = dct_blocks.shape

        # Reconstruir imagen desde DCT
        reconstructed = np.zeros((ny * 8, nx * 8))

        for i in range(ny):
            for j in range(nx):
                block = fftpack.idct(
                    fftpack.idct(dct_blocks[i, j].T, norm="ortho").T, norm="ortho"
                )
                reconstructed[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = block + 0.5

        # Medir discontinuidades en bordes de bloques
        h_edges = np.abs(reconstructed[:, 7::8] - reconstructed[:, 8::8])
        # Nota: revisar indices si cruzan limites.
        # 7::8 toma pixel 7, 15, 23... (ultimo de bloque)
        # 8::8 toma pixel 8, 16, 24... (primero de siguiente bloque)
        # Si image width es exacto, len(7::8) == len(8::8) excepto si termina justo?
        # Si W = 16, 7::8 -> [7], 8::8 -> [8]. diff -> [abs(img[7]-img[8])]
        # Si W = 8, 7::8 -> [7], 8::8 -> []. Mismatch.
        # Fix slice size match
        min_h = min(h_edges.shape[1], reconstructed[:, 8::8].shape[1])
        if min_h == 0:
            h_mean = 0
        else:
            h_edges = np.abs(
                reconstructed[:, 7::8][:, :min_h] - reconstructed[:, 8::8][:, :min_h]
            )
            h_mean = h_edges.mean()

        v_edges = np.abs(reconstructed[7::8, :] - reconstructed[8::8, :])
        min_v = min(v_edges.shape[0], reconstructed[8::8, :].shape[0])
        if min_v == 0:
            v_mean = 0
        else:
            v_edges = np.abs(
                reconstructed[7::8, :][:min_v, :] - reconstructed[8::8, :][:min_v, :]
            )
            v_mean = v_edges.mean()

        blockiness = (h_mean + v_mean) / 2

        return blockiness

    def analyze(self, image: np.ndarray) -> dict:
        """
        Pipeline completo de análisis DCT.
        """
        dct_blocks = self.compute_dct_blocks(image)
        features = self.extract_coefficient_statistics(dct_blocks)

        return {"dct_blocks": dct_blocks, "features": features}
