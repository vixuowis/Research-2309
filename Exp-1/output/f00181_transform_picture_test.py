from f00181_transform_picture import *
import pytest
from PIL import Image

@pytest.fixture

def test_transform_picture():
    image = transform_picture('https://example.com/picture.jpg')
    assert isinstance(image, Image.Image)
    assert image.width == 1000
    assert image.height == 800
    assert (100, 100) in image
    assert (200, 200) not in image
