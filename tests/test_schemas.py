import pytest
from pydantic import ValidationError

from app.schemas import PredictionRequest

VALID = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}


def test_valid_payload_is_accepted():
    request = PredictionRequest(**VALID)
    assert request.to_features() == [5.1, 3.5, 1.4, 0.2]


@pytest.mark.parametrize("missing", list(VALID))
def test_missing_field_is_rejected(missing):
    payload = {key: value for key, value in VALID.items() if key != missing}
    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


@pytest.mark.parametrize("value", [0, -1.5, 10.1])
def test_out_of_range_value_is_rejected(value):
    with pytest.raises(ValidationError):
        PredictionRequest(**{**VALID, "sepal_length": value})


def test_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        PredictionRequest(**VALID, color="red")
