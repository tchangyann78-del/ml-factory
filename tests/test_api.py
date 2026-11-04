import pytest

VALID = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        ((5.1, 3.5, 1.4, 0.2), "setosa"),
        ((5.7, 2.8, 4.1, 1.3), "versicolor"),
        ((6.3, 3.3, 6.0, 2.5), "virginica"),
    ],
)
def test_predict_returns_expected_species(client, features, expected):
    payload = dict(zip(VALID, features, strict=True))
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == expected
    assert 0.0 <= body["confidence"] <= 1.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("sepal_length", -1), ("petal_width", 0), ("petal_length", 99), ("sepal_width", "abc")],
)
def test_predict_rejects_invalid_values_with_422(client, field, value):
    response = client.post("/predict", json={**VALID, field: value})
    assert response.status_code == 422


def test_predict_rejects_missing_field_with_422(client):
    payload = {key: value for key, value in VALID.items() if key != "petal_width"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "petal_width"]


def test_predict_rejects_unknown_field_with_422(client):
    response = client.post("/predict", json={**VALID, "color": "blue"})
    assert response.status_code == 422


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_endpoints_return_503_when_model_is_missing(client, monkeypatch):
    monkeypatch.setattr(client.app.state, "model", None)
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=VALID).status_code == 503
