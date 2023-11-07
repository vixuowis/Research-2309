from f00075_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained("distilbert-base-uncased")
    assert isinstance(model, TFAutoModelForTokenClassification)


test_from_pretrained()
