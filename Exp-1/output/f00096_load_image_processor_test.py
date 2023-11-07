from f00096_load_image_processor import *
def test_load_image_processor():
    model_name = "google/vit-base-patch16-224"
    image_processor = load_image_processor(model_name)
    assert isinstance(image_processor, AutoImageProcessor)

test_load_image_processor()
