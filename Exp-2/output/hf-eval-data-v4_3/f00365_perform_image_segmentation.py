# requirements_file --------------------

import subprocess

requirements = ["transformers", "Pillow", "matplotlib", "requests"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import requests

# function_code --------------------

def perform_image_segmentation(image_path: str) -> Image:
    """
    Segments clothes in a given image using a pre-trained SegFormer model.

    Args:
        image_path (str): The path to the image file or a URL of the image.

    Returns:
        Image: The segmented image.

    Raises:
        ValueError: If image_path is not accessible.
    """
    try:
        if image_path.startswith('http://') or image_path.startswith('https://'):
            response = requests.get(image_path, stream=True)
            response.raise_for_status()
            image = Image.open(response.raw)
        else:
            image = Image.open(image_path)
    except Exception as e:
        raise ValueError(f'Failed to load image: {e}')

    model = SegformerForSemanticSegmentation.from_pretrained('mattmdjaga/segformer_b2_clothes')
    extractor = AutoFeatureExtractor.from_pretrained('mattmdjaga/segformer_b2_clothes')

    inputs = extractor(images=image, return_tensors='pt')
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu()
    upsampled_logits = F.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0]

    plt.imshow(pred_seg)
    plt.axis('off')
    plt.show()

    return pred_seg

# test_function_code --------------------

def test_perform_image_segmentation():
    print('Testing started.')
    
    # Test case 1: Test with a valid URL
    print('Testing case [1/3] started.')
    image_url = 'https://example.com/test_image.jpg'
    try:
        segmented_image = perform_image_segmentation(image_url)
        assert isinstance(segmented_image, Image.Image), 'Segmented image is not an instance of PIL.Image.Image.'
    except Exception as e:
        assert False, f'Test case [1/3] failed: {e}'

    # Test case 2: Test with an invalid URL
    print('Testing case [2/3] started.')
    invalid_image_url = 'https://example.com/non_existent_image.jpg'
    try:
        perform_image_segmentation(invalid_image_url)
        assert False, 'Test case [2/3] failed: Invalid image URL did not raise an error.'
    except ValueError:
        assert True
    except Exception as e:
        assert False, f'Test case [2/3] failed with an unexpected error: {e}'

    # Test case 3: Test with a local image path
    print('Testing case [3/3] started.')
    local_image_path = 'path/to/local_image.jpg'
    try:
        segmented_image = perform_image_segmentation(local_image_path)
        assert isinstance(segmented_image, Image.Image), 'Segmented image is not an instance of PIL.Image.Image.'
    except Exception as e:
        assert False, f'Test case [3/3] failed: {e}'

    print('Testing finished.')

# call_test_function_line --------------------

test_perform_image_segmentation()