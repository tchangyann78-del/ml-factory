"""Chargement et inférence du modèle ML (isolés de la couche HTTP)."""

import os
from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.joblib"


class IrisModel:
    """Enveloppe fine autour de l'estimateur scikit-learn."""

    def __init__(self, estimator, target_names: list[str], version: str) -> None:
        self._estimator = estimator
        self._target_names = target_names
        self.version = version

    def predict(self, features: list[float]) -> tuple[str, float]:
        probabilities = self._estimator.predict_proba([features])[0]
        best = int(probabilities.argmax())
        return self._target_names[best], round(float(probabilities[best]), 4)


def load_model(path: Path | None = None) -> IrisModel:
    """Charge l'artefact. Lève FileNotFoundError s'il est absent (fail-fast)."""
    model_path = Path(path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    bundle = joblib.load(model_path)
    return IrisModel(bundle["model"], bundle["target_names"], bundle["version"])
