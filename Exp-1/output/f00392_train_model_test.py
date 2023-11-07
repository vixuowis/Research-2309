from f00392_train_model import *
def test_train_model():
    model = create_model()
    train_data = generate_train_data()
    test_data = generate_test_data()
    epochs = 3
    callbacks = [callback1, callback2]

    train_model(model, train_data, test_data, epochs, callbacks)

    # Assert statements for testing
    assert model.trained
    assert model.accuracy > 0.9


test_train_model()
