from f00323_train_model import *
def test_train_model():
    model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")
    trained_model = train_model(model)
    assert isinstance(trained_model, AutoModelForMaskedLM)
    print("Train model test passed!")
