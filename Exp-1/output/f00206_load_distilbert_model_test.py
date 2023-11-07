from f00206_load_distilbert_model import *
def test_load_distilbert_model():
	model = load_distilbert_model(2, id2label, label2id)
	assert isinstance(model, TFAutoModelForSequenceClassification)

id2label = {0: "negative", 1: "positive"}
label2id = {"negative": 0, "positive": 1}

test_load_distilbert_model()
