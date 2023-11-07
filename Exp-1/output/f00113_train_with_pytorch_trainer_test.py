from f00113_train_with_pytorch_trainer import *
def test_train_with_pytorch_trainer():
    model = train_with_pytorch_trainer()
    assert isinstance(model, AutoModelForSequenceClassification)
    assert model.config.num_labels == 5
