from f00886_initialize_model import *
def test_initialize_model():
    model = initialize_model(num_aggregation_labels=3, average_logits_per_cell=True)
    assert isinstance(model, TapasForQuestionAnswering)
    assert model.config.num_aggregation_labels == 3
    assert model.config.average_logits_per_cell == True

test_initialize_model()
