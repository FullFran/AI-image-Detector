#!/usr/bin/env python3
"""
Script de entrenamiento del detector de imágenes IA con métricas.
Entrena con el dataset Women in AI 2025 (SDXL) y genera un informe de evaluación.
"""

import glob
import os
import pickle
import sys
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Añadir el path del paquete
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class GradientCovarianceDetector:
    """Detector de imágenes IA basado en covarianza de gradientes."""

    LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])

    def rgb_to_luminance(self, image: np.ndarray) -> np.ndarray:
        if image.max() > 1:
            image = image.astype(np.float64) / 255.0
        return np.dot(image[..., :3], self.LUMINANCE_WEIGHTS)

    def compute_gradients(self, luminance: np.ndarray) -> tuple:
        Gx = ndimage.sobel(luminance, axis=1, mode="reflect")
        Gy = ndimage.sobel(luminance, axis=0, mode="reflect")
        return Gx, Gy

    def compute_covariance_matrix(self, Gx: np.ndarray, Gy: np.ndarray) -> np.ndarray:
        M = np.column_stack([Gx.flatten(), Gy.flatten()])
        M_centered = M - M.mean(axis=0)
        return (1 / M.shape[0]) * (M_centered.T @ M_centered)

    def extract_features(self, image: np.ndarray) -> dict:
        L = self.rgb_to_luminance(image)
        Gx, Gy = self.compute_gradients(L)
        C = self.compute_covariance_matrix(Gx, Gy)

        eigenvalues, _ = np.linalg.eigh(C)
        eigenvalues = np.sort(eigenvalues)[::-1]
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


def load_images_from_folder(folder: str, label: int, max_images: int = None):
    """Carga imágenes de una carpeta y extrae features."""
    detector = GradientCovarianceDetector()
    features_list = []
    labels = []

    extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(folder, ext)))

    if max_images:
        image_files = image_files[:max_images]

    print(f"  Procesando {len(image_files)} imágenes de {folder}...")

    for i, img_path in enumerate(image_files):
        try:
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = np.array(img)

            features = detector.extract_features(img_array)
            features_list.append(list(features.values()))
            labels.append(label)

            if (i + 1) % 50 == 0:
                print(f"    Procesadas {i + 1}/{len(image_files)} imágenes")

        except Exception as e:
            print(f"    Error procesando {img_path}: {e}")
            continue

    return features_list, labels


def main():
    print("=" * 70)
    print("🔬 ENTRENAMIENTO DEL DETECTOR DE IMÁGENES IA")
    print("   Dataset: Women in AI 2025 (SDXL)")
    print("=" * 70)

    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "train")
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    model_path = os.path.join(
        os.path.dirname(__file__), "..", "models", "classifier.pkl"
    )

    # Verificar que existen las carpetas
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"❌ Error: No se encuentran las carpetas {real_dir} y {fake_dir}")
        return

    # Cargar imágenes
    print("\n📂 Cargando imágenes...")
    real_features, real_labels = load_images_from_folder(real_dir, label=0)
    fake_features, fake_labels = load_images_from_folder(fake_dir, label=1)

    # Combinar
    X = np.array(real_features + fake_features)
    y = np.array(real_labels + fake_labels)

    print(f"\n📊 Dataset total: {len(X)} imágenes")
    print(f"   - Reales: {len(real_features)}")
    print(f"   - Fake (IA): {len(fake_features)}")

    # Dividir train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n🔄 División del dataset:")
    print(f"   - Train: {len(X_train)} imágenes")
    print(f"   - Test: {len(X_test)} imágenes")

    # Entrenar modelo
    print("\n🎓 Entrenando Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Predecir
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Métricas
    print("\n" + "=" * 70)
    print("📈 MÉTRICAS DE EVALUACIÓN")
    print("=" * 70)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n   Accuracy:  {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n📊 Matriz de Confusión:")
    print("                    Predicho")
    print("                  REAL    FAKE")
    print(f"   Real REAL    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"        FAKE    {cm[1][0]:5d}   {cm[1][1]:5d}")

    # Classification Report
    print("\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

    # Guardar modelo
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\n💾 Modelo guardado en: {model_path}")

    # Feature importance
    feature_names = [
        "lambda1",
        "lambda2",
        "trace",
        "determinant",
        "eigenvalue_ratio",
        "anisotropy",
        "eccentricity",
        "frobenius_norm",
        "condition_number",
        "covariance_xy",
        "variance_x",
        "variance_y",
        "differential_entropy",
    ]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n🔍 Importancia de Features (Top 5):")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"   {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("=" * 70)


if __name__ == "__main__":
    main()
