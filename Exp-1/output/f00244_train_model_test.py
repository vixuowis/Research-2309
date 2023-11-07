from f00244_train_model import *
def test_train_model():
    model = keras.Sequential()
    train_data = tf.data.Dataset()
    val_data = tf.data.Dataset()
    num_epochs = 3
    callbacks = []
    train_model(model, train_data, val_data, num_epochs, callbacks)

def test_train_model()
    assert True
