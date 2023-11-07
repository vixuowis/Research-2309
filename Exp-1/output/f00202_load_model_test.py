from f00202_load_model import *
def test_load_model():
    num_labels = 2
    id2label = {0: "negative", 1: "positive"}
    label2id = {"negative": 0, "positive": 1}
    model = load_model(num_labels, id2label, label2id)
    assert isinstance(model, AutoModelForSequenceClassification)

test_load_model()
