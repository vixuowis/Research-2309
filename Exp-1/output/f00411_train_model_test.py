from f00411_train_model import *
def test_train_model():
    train_model()
    assert os.path.exists("./results")
    assert os.path.exists("./logs")
