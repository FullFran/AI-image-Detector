import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ai_detector.data.generation import generate_training_dataset
from ai_detector.features.gradient import GradientCovarianceDetector
from ai_detector.utils import load_image_from_path


class ClassificationService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClassificationService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("Initializing ClassificationService... Training model...")
        self.detector = GradientCovarianceDetector()
        self.model = self._train_model()
        print("Model trained.")

    def _train_model(self):
        # Intentar cargar datos reales desde data/train
        # Buscamos la carpeta data relativa a la raiz del proyecto
        # Estructura: root/src/app/services/classification.py -> subimos 4 niveles para root
        root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../")
        )
        real_dir = os.path.join(root_dir, "data/train/real")
        fake_dir = os.path.join(root_dir, "data/train/fake")

        real_images = []
        fake_images = []

        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]

        if os.path.exists(real_dir) and os.path.exists(fake_dir):
            for ext in extensions:
                real_images.extend(glob.glob(os.path.join(real_dir, ext)))
                fake_images.extend(glob.glob(os.path.join(fake_dir, ext)))

        print(f"Checking for user data at {real_dir} and {fake_dir}...")
        print(
            f"Found {len(real_images)} real images and {len(fake_images)} fake images."
        )

        df_train = None

        # Si tenemos suficientes datos reales, entrenamos con ellos
        if len(real_images) >= 5 and len(fake_images) >= 5:
            print(" Sufficient real data found. Training with USER PROVIDED images.")
            data = []

            def process_files(files, label):
                count = 0
                for fpath in files:
                    try:
                        img = load_image_from_path(fpath)
                        res = self.detector.analyze(img)
                        feats = res["features"].copy()
                        feats["label"] = label
                        data.append(feats)
                        count += 1
                    except Exception as e:
                        print(f"Error loading {fpath}: {e}")
                return count

            n_real = process_files(real_images, 0)  # 0 = REAL
            n_fake = process_files(fake_images, 1)  # 1 = FAKE/IA

            if n_real > 0 and n_fake > 0:
                df_train = pd.DataFrame(data)
                print(f"Training set created with {len(df_train)} samples.")

        # Fallback a sintético si no hay suficientes datos
        if df_train is None:
            print(
                "CAUTION: Using SYNTHETIC data for training (1/f noise vs artifacts)."
            )
            print(
                "This is for demonstration only. For real detection, populate data/train/real and data/train/fake."
            )
            df_train = generate_training_dataset(n_samples=100)

        feature_cols = [c for c in df_train.columns if c != "label"]
        X_train = df_train[feature_cols].values
        y_train = df_train["label"].values

        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        clf.fit(X_train, y_train)
        self.feature_cols = feature_cols
        return clf

    def predict(self, image: np.ndarray) -> dict:
        analysis = self.detector.analyze(image)
        features = analysis["features"]

        X_test = np.array([[features[col] for col in self.feature_cols]])

        prediction_idx = self.model.predict(X_test)[0]
        probabilities = self.model.predict_proba(X_test)[0]

        return {
            "prediction": "REAL" if prediction_idx == 0 else "IA/FAKE",
            "probability_real": probabilities[0],
            "probability_fake": probabilities[1],
            "features": features,
        }

    def retrain(self):
        """Re-entrena el modelo con los datos actuales en data/train/."""
        print("Re-training model with current data...")
        self.model = self._train_model()
        print("Model re-trained successfully.")
        return True

    def train_from_arrays(self, real_images: list, fake_images: list) -> dict:
        """
        Entrena el modelo directamente desde arrays de imágenes en memoria.
        real_images: lista de arrays numpy (imágenes reales)
        fake_images: lista de arrays numpy (imágenes IA/fake)
        """
        if len(real_images) < 1 or len(fake_images) < 1:
            return {
                "success": False,
                "error": "Se necesita al menos 1 imagen de cada clase.",
            }

        data = []

        # Procesar reales
        for img in real_images:
            try:
                res = self.detector.analyze(img)
                feats = res["features"].copy()
                feats["label"] = 0  # REAL
                data.append(feats)
            except Exception as e:
                print(f"Error procesando imagen real: {e}")

        # Procesar fakes
        for img in fake_images:
            try:
                res = self.detector.analyze(img)
                feats = res["features"].copy()
                feats["label"] = 1  # FAKE
                data.append(feats)
            except Exception as e:
                print(f"Error procesando imagen fake: {e}")

        if len(data) < 2:
            return {
                "success": False,
                "error": "No se pudieron procesar suficientes imágenes.",
            }

        df_train = pd.DataFrame(data)
        feature_cols = [c for c in df_train.columns if c != "label"]
        X_train = df_train[feature_cols].values
        y_train = df_train["label"].values

        # Verificar que hay ambas clases
        if len(set(y_train)) < 2:
            return {
                "success": False,
                "error": "Se necesitan imágenes de ambas clases (real y fake).",
            }

        clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        clf.fit(X_train, y_train)

        self.model = clf
        self.feature_cols = feature_cols

        return {
            "success": True,
            "n_real": int(sum(y_train == 0)),
            "n_fake": int(sum(y_train == 1)),
            "total_samples": len(y_train),
        }
