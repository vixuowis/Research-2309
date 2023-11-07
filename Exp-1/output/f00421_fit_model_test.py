from f00421_fit_model import *
def test_fit_model():
    model = tf.keras.Sequential()
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    num_epochs = 2
    callbacks = []
    fit_model(model, train_data, validation_data, num_epochs, callbacks)


test_fit_model()
