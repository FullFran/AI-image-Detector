import io

import matplotlib.pyplot as plt
import numpy as np


def visualize_analysis(image: np.ndarray, results: dict, title: str = "Análisis"):
    """Visualiza el análisis de una imagen."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Imagen Original")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(results["luminance"], cmap="gray")
    axes[0, 1].set_title("Luminancia L(x,y)")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    grad_magnitude = np.sqrt(results["gradient_x"] ** 2 + results["gradient_y"] ** 2)
    im2 = axes[0, 2].imshow(grad_magnitude, cmap="hot")
    axes[0, 2].set_title("|∇L| = √(Gx² + Gy²)")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(results["gradient_x"], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1, 0].set_title("Gradiente Gx = ∂L/∂x")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(results["gradient_y"], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axes[1, 1].set_title("Gradiente Gy = ∂L/∂y")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    M = results["gradient_matrix"]
    n_samples = min(5000, M.shape[0])
    idx = np.random.choice(M.shape[0], n_samples, replace=False)

    axes[1, 2].scatter(M[idx, 0], M[idx, 1], alpha=0.1, s=1, c="blue")

    C = results["covariance_matrix"]
    eigenvalues, eigenvectors = np.linalg.eigh(C)

    theta = np.linspace(0, 2 * np.pi, 100)
    scale = 2
    ellipse = (
        scale
        * np.sqrt(np.abs(eigenvalues))
        * np.column_stack([np.cos(theta), np.sin(theta)])
    )
    ellipse = ellipse @ eigenvectors.T

    axes[1, 2].plot(ellipse[:, 0], ellipse[:, 1], "r-", linewidth=2, label="Elipse 2σ")
    axes[1, 2].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 2].axvline(x=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 2].set_xlabel("Gx")
    axes[1, 2].set_ylabel("Gy")
    axes[1, 2].set_title("Distribución de Gradientes")
    axes[1, 2].set_aspect("equal")
    axes[1, 2].legend()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def visualize_unified_analysis(
    image: np.ndarray, detector, title: str = "Análisis Espectral y DCT"
):
    """
    Visualización completa del análisis espectral, DCT y color.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    # Imagen original
    if image.max() > 1:
        img_display = image / 255.0
    else:
        img_display = image
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    # Espectro de potencia (log)
    fft = detector.spectral_analyzer.compute_fft(image)
    power = detector.spectral_analyzer.compute_power_spectrum(fft)
    log_power = np.log(power + 1)
    axes[0, 1].imshow(log_power, cmap="viridis")
    axes[0, 1].set_title("log(Espectro de Potencia)")
    axes[0, 1].axis("off")

    # Perfil radial
    freqs, profile = detector.spectral_analyzer.azimuthal_average(power)
    valid = freqs > 0
    axes[0, 2].loglog(freqs[valid], profile[valid], "b-", linewidth=1.5)

    # Ajuste ley de potencias
    power_law = detector.spectral_analyzer.fit_power_law(freqs, profile)
    if not np.isnan(power_law["alpha"]):
        f_fit = freqs[freqs > 1]
        p_fit = power_law["A"] / (f_fit ** power_law["alpha"])
        axes[0, 2].loglog(
            f_fit, p_fit, "r--", linewidth=2, label=f"α = {power_law['alpha']:.2f}"
        )
    axes[0, 2].set_xlabel("Frecuencia")
    axes[0, 2].set_ylabel("Potencia")
    axes[0, 2].set_title("Perfil Radial (log-log)")
    axes[0, 2].legend()

    # DCT promedio
    dct_blocks = detector.dct_analyzer.compute_dct_blocks(image)
    # dct_blocks shape: (ny, nx, 8, 8)
    # mean over blocks (axis 0, 1) to get average 8x8 block
    avg_dct = np.abs(dct_blocks).mean(axis=(0, 1))
    im = axes[0, 3].imshow(np.log(avg_dct + 1e-6), cmap="hot")
    axes[0, 3].set_title("log(|DCT| promedio)")
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)

    # Histograma de coeficientes AC
    # dct_blocks[:, :, 1:, :]? No, AC excludes only 0,0.
    # But usually visualized excluding DC.
    # Original notebook: ac_coeffs = dct_blocks[:, :, 1:, :].flatten() -> This drops row 0 of every block (8 coeffs).
    # That captures many AC coeffs but not all (keeps 0,1..0,7 but drops 1,0..7,0?).
    # Wait, block shape is 8x8. dct_blocks[:,:, 1:, :] slicies indices 2 and 3 (the 8x8 block).
    # 1: means rows 1-7. So we drop row 0. Row 0 contains DC (0,0) and low horizontal freq.
    # It's an approximation. Let's keep consistent with original notebook logic.
    ac_coeffs = dct_blocks[:, :, 1:, :].flatten()
    axes[1, 0].hist(ac_coeffs, bins=100, density=True, alpha=0.7, color="steelblue")
    axes[1, 0].set_xlabel("Valor del coeficiente")
    axes[1, 0].set_ylabel("Densidad")
    axes[1, 0].set_title("Histograma coeficientes AC (parcial)")
    axes[1, 0].set_xlim(-0.5, 0.5)

    # Canales de color YCbCr
    ycbcr = detector.color_analyzer.rgb_to_ycbcr(image)
    axes[1, 1].imshow(ycbcr[..., 1], cmap="RdBu_r", vmin=0, vmax=1)
    axes[1, 1].set_title("Canal Cb")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(ycbcr[..., 2], cmap="RdBu_r", vmin=0, vmax=1)
    axes[1, 2].set_title("Canal Cr")
    axes[1, 2].axis("off")

    # Saturación HSV
    hsv = detector.color_analyzer.rgb_to_hsv(image)
    axes[1, 3].imshow(hsv[..., 1], cmap="plasma")
    axes[1, 3].set_title("Saturación (HSV)")
    axes[1, 3].axis("off")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()


def create_report_plot(image: np.ndarray, results: dict, title: str) -> io.BytesIO:
    """
    Genera un grid de visualización con Imagen + Mapa de Gradientes y devuelve el buffer.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # 1. Imagen Original
    if image.max() > 1:
        img_display = image.astype(np.float64) / 255.0
    else:
        img_display = image

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # 2. Mapa de Gradientes (Luminance Gradient Magnitude)
    # Calculamos magnitud: |∇L| = sqrt(Gx^2 + Gy^2)
    grad_x = results.get("gradient_x")
    grad_y = results.get("gradient_y")

    if grad_x is not None and grad_y is not None:
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Normalizar para visualización clara
        # Usamos escala logarítmica o clipping para resaltar detalles
        magnitude = np.clip(magnitude, 0, np.percentile(magnitude, 99))

        im = axes[1].imshow(magnitude, cmap="gray")
        axes[1].set_title("Luminance Gradient Map", fontsize=14)
        axes[1].axis("off")

    plt.suptitle(
        title,
        fontsize=20,
        fontweight="bold",
        color="darkred" if "FAKE" in title else "darkgreen",
    )
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf
