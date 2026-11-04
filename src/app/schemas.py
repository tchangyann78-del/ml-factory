"""Contrats d'API : ce que le service accepte et ce qu'il renvoie."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Mesures d'une fleur d'iris, en centimètres."""

    # extra="forbid" : un champ inconnu (faute de frappe) est rejeté au lieu d'être ignoré.
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        },
    )

    sepal_length: float = Field(gt=0, le=10, description="Longueur du sépale (cm)")
    sepal_width: float = Field(gt=0, le=10, description="Largeur du sépale (cm)")
    petal_length: float = Field(gt=0, le=10, description="Longueur du pétale (cm)")
    petal_width: float = Field(gt=0, le=10, description="Largeur du pétale (cm)")

    def to_features(self) -> list[float]:
        """Vecteur de caractéristiques, dans l'ordre attendu par le modèle."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictionResponse(BaseModel):
    species: Literal["setosa", "versicolor", "virginica"]
    confidence: float = Field(ge=0, le=1, description="Probabilité de la classe prédite")
    version: str = Field(description="Version du modèle qui a produit la prédiction")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    ready: bool
    version: str
