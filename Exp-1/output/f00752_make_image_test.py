from f00752_make_image import *
def test_make_image():
    image = make_image()
    assert isinstance(image, Image.Image)
    assert image.size == (500, 500)
    # Add more test cases if needed

test_make_image()
