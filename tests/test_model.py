from app.model import load_model


def test_model_predicts_setosa_with_high_confidence():
    model = load_model()
    species, confidence = model.predict([5.1, 3.5, 1.4, 0.2])
    assert species == "setosa"
    assert 0.5 < confidence <= 1.0
