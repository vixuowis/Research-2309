from f00072_AutoModelForSequenceClassification.from_pretrained import *
def test_from_pretrained():
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    assert isinstance(model, AutoModelForSequenceClassification)

test_from_pretrained()
