from f00569_load_image import *
def test_load_image():
    assert isinstance(load_image(), Image.Image)


def test_all():
    test_load_image()
