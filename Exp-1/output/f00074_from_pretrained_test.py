from f00074_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained("distilbert-base-uncased")
    assert isinstance(model, TFAutoModelForSequenceClassification)

test_from_pretrained()
