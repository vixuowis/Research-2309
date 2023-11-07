from f00763_upscale_image import *
def test_upscale_image():
    # Test case 1
    image = Image.open('input.jpg')
    upscaled_image = upscale_image(image)
    assert upscaled_image.size == (1024, 1024)

    # Test case 2
    image = Image.open('input.png')
    upscaled_image = upscale_image(image)
    assert upscaled_image.size == (1024, 1024)

    # Test case 3
    image = Image.open('input.bmp')
    upscaled_image = upscale_image(image)
    assert upscaled_image.size == (1024, 1024)

    # Test case 4
    image = Image.open('input.gif')
    upscaled_image = upscale_image(image)
    assert upscaled_image.size == (1024, 1024)

    # Test case 5
    image = Image.open('input.tiff')
    upscaled_image = upscale_image(image)
    assert upscaled_image.size == (1024, 1024)

    print('All test cases pass')

test_upscale_image()
