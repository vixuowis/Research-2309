from f00764_load_tf_model import *
def test_load_tf_model():
    model_path = "some_folder"
    model = load_tf_model(model_path)
    assert isinstance(model, TFPreTrainedModel)


test_load_tf_model()
