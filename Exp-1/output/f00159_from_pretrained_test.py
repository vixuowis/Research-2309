from f00159_from_pretrained import *
def test_from_pretrained():
    model_path = "path/to/awesome-name-you-picked"
    flax_model = from_pretrained(model_path, from_pt=True)
    assert isinstance(flax_model, FlaxDistilBertForSequenceClassification)

test_from_pretrained()
