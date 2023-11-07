from f00480_load_image_processor import *
def test_load_image_processor():
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = load_image_processor(checkpoint)
    assert isinstance(image_processor, AutoImageProcessor)


test_load_image_processor()
