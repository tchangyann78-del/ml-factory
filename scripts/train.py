"""Entraîne un petit classifieur Iris et sérialise l'artefact dans models/."""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

OUTPUT = Path(__file__).resolve().parents[1] / "models" / "model.joblib"
MODEL_VERSION = "1.0.0"


def main() -> None:
    iris = load_iris()
    estimator = LogisticRegression(max_iter=200)
    estimator.fit(iris.data, iris.target)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": estimator,
        "target_names": [str(name) for name in iris.target_names],
        "version": MODEL_VERSION,
    }
    joblib.dump(bundle, OUTPUT)
    print(f"Modèle sauvegardé dans {OUTPUT}")


if __name__ == "__main__":
    main()
