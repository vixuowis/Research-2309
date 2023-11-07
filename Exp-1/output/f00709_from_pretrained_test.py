from f00709_from_pretrained import *
def test_from_pretrained():
    model = from_pretrained("distilbert-base-uncased")
    assert isinstance(model, DistilBertForQuestionAnswering)

test_from_pretrained()
