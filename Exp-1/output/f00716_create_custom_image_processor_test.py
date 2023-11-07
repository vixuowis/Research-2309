from f00716_create_custom_image_processor import *
def test_create_custom_image_processor():
    assert isinstance(create_custom_image_processor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3]), ViTImageProcessor)
    assert create_custom_image_processor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3]).resample == "PIL.Image.BOX"
    assert not create_custom_image_processor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3]).do_normalize
    assert create_custom_image_processor(resample="PIL.Image.BOX", do_normalize=False, image_mean=[0.3, 0.3, 0.3]).image_mean == [0.3, 0.3, 0.3]

def test_entry():
    test_create_custom_image_processor()

test_entry()
