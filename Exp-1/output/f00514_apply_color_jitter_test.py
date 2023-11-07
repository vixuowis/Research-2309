from f00514_apply_color_jitter import *
def test_apply_color_jitter():
    import numpy as np
    from PIL import Image
    
    # Test case 1
    image = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    jittered_image = apply_color_jitter(image)
    assert isinstance(jittered_image, np.ndarray)
    assert jittered_image.shape == (32, 32, 3)
    
    # Test case 2
    image = Image.fromarray(np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8))
    jittered_image = apply_color_jitter(image, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
    assert isinstance(jittered_image, Image.Image)
    
    # Test case 3
    image = np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    jittered_image = apply_color_jitter(image, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
    assert isinstance(jittered_image, np.ndarray)
    assert jittered_image.shape == (64, 64, 3)
    
    # Test case 4
    image = Image.fromarray(np.random.randint(0, 255, size=(64, 64, 3), dtype=np.uint8))
    jittered_image = apply_color_jitter(image)
    assert isinstance(jittered_image, Image.Image)
    
    # Test case 5
    image = np.random.randint(0, 255, size=(128, 128, 3), dtype=np.uint8)
    jittered_image = apply_color_jitter(image, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)
    assert isinstance(jittered_image, np.ndarray)
    assert jittered_image.shape == (128, 128, 3)
    
    print('All test cases pass')

test_apply_color_jitter()
