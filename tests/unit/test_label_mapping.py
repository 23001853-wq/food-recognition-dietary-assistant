import pytest

from food_assistant.domain.schemas import Prediction


@pytest.mark.parametrize("patch", [{"class_id": -1}, {"score": float("nan")}, {"score": 1.1}])
def test_prediction_contract_rejects_invalid_indices_and_scores(patch):
    with pytest.raises(ValueError):
        Prediction(
            **(
                {"class_id": 0, "class_key": "pizza", "display_name_vi": "Pizza", "score": 0.5}
                | patch
            )
        )
