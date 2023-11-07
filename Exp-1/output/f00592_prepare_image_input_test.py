from f00592_prepare_image_input import *
def test_prepare_image_input():
    
    image = Image.open("path/to/image.jpg")
    image_processor = ImageProcessor()
    
    pixel_values = prepare_image_input(image, image_processor)
    
    assert isinstance(pixel_values, torch.Tensor)
    assert pixel_values.shape == (1, 3, 224, 224)
    
    print("All test cases pass")
