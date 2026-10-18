"""Point d'entrée de l'API de prédiction."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, status

from app import __version__
from app.model import IrisModel, load_model
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Le modèle est chargé UNE seule fois, au démarrage (pas à chaque requête).
    # S'il est introuvable, le process s'arrête immédiatement : fail-fast.
    app.state.model = load_model()
    logger.info("Modèle chargé (version %s)", app.state.model.version)
    yield
    app.state.model = None


app = FastAPI(
    title="Prediction Service",
    description="Microservice d'inférence (classification Iris) - Projet MLOps 1",
    version=__version__,
    lifespan=lifespan,
)


def get_model(request: Request) -> IrisModel:
    """Dépendance : fournit le modèle chargé, ou répond 503 s'il est indisponible."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle indisponible",
        )
    return model


ModelDep = Annotated[IrisModel, Depends(get_model)]


@app.get(
    "/health",
    tags=["ops"],
    summary="Healthcheck (readiness)",
    responses={503: {"description": "Le modèle n'est pas chargé"}},
)
def health(_model: ModelDep) -> HealthResponse:
    return HealthResponse(ready=True, version=__version__)


@app.post(
    "/predict",
    tags=["inference"],
    summary="Prédit l'espèce d'iris à partir de 4 mesures",
    responses={503: {"description": "Le modèle n'est pas chargé"}},
)
def predict(payload: PredictionRequest, model: ModelDep) -> PredictionResponse:
    # `def` (et non `async def`) : l'inférence est CPU-bound. FastAPI l'exécute dans
    # un thread pool, l'event loop reste donc disponible pour les autres requêtes.
    species, confidence = model.predict(payload.to_features())
    logger.info("prediction species=%s confidence=%.3f", species, confidence)
    return PredictionResponse(species=species, confidence=confidence, version=model.version)
