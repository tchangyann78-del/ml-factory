import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Le `with` déclenche le lifespan : le modèle est donc chargé comme en vrai.
    with TestClient(app) as test_client:
        yield test_client
