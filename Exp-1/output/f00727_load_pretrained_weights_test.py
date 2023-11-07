from f00727_load_pretrained_weights import *
def test_load_pretrained_weights():
    model = MyModel()
    load_pretrained_weights(model)
    assert model.pretrained
    assert model.weights_loaded
    assert model.weights.shape == (1000,)

    model2 = MyModel()
    load_pretrained_weights(model2)
    assert model.weights_loaded
    assert model.weights_equal(model2.weights)
