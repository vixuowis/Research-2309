from f00755_create_image import *
def test_create_image():
    assert create_image('A house and car') == 0
    assert create_image('Another image') == 0
    assert create_image('Image 123') == 0
