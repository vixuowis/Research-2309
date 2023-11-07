from f00531_load_image import *
def test_load_image():
    image_path = 'path/to/image.jpg'
    image = load_image(image_path)
    assert isinstance(image, np.ndarray)
    assert image.shape == (height, width, channels)
    # Additional test cases
    # ...
    print('All test cases pass')

test_load_image()
