from f00152_train_model import *
import pytest

@pytest.mark.parametrize(
    "model",
    [
        (model1),
        (model2),
        (model3),
        (model4),
        (model5),
    ],
)
def test_train_model(model):
    assert train_model(model) == None
