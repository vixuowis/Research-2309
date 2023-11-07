from f00505_load_image_processor import *
def test_load_image_processor():
    image = ...  # input image
    model_name = "MariaK/food_classifier"
    inputs = load_image_processor(model_name)
    assert inputs is not None

test_load_image_processor()
