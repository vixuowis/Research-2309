from f00627_train_model import *
def test_train_model():
    model_checkpoint = "path/to/model_checkpoint"
    id2label = {0: "label1", 1: "label2", 2: "label3"}
    label2id = {"label1": 0, "label2": 1, "label3": 2}

    model = train_model(model_checkpoint, id2label, label2id)

    assert isinstance(model, ViltForQuestionAnswering)


test_train_model()
