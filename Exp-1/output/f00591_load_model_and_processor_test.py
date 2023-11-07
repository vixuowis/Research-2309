from f00591_load_model_and_processor import *
def test_load_model_and_processor():
    checkpoint = "vinvino02/glpn-nyu"
    image_processor, model = load_model_and_processor(checkpoint)
    assert isinstance(image_processor, AutoImageProcessor)
    assert isinstance(model, AutoModelForDepthEstimation)


test_load_model_and_processor()
