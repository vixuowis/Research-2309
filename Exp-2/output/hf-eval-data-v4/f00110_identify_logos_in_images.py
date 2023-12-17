# requirements_file --------------------

!pip install -U timm, Pillow

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def identify_logos_in_images(image_url):
    # Load an image from the provided URL
    img = Image.open(urlopen(image_url))

    # Create a ConvNeXt-V2 model pretrained on ImageNet-1k dataset
    model = timm.create_model('convnextv2_huge.fcmae_ft_in1k', pretrained=True)
    # Set the model to evaluation mode
    model = model.eval()

    # Configure data preprocessing based on model's requirement
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Preprocess the image and make a classification prediction
    output = model(transforms(img).unsqueeze(0))

    # Define the indices for logo classes
    logo_class_indices = [0, 1, 2]  # Replace with actual indices corresponding to logos
    # Calculate the sum of softmax probabilities for logo classes
    logo_score = output.softmax(dim=1)[0, logo_class_indices].sum().item()

    # Determine if a logo is present based on the score threshold
    is_logo_present = logo_score > 0.5
    return is_logo_present

# test_function_code --------------------

def test_identify_logos_in_images():
    print('Testing started.')
    test_image_urls = {
        'logo': 'URL to an image with a logo',
        'no_logo': 'URL to an image without a logo'
    }

    # Test case 1: Image with a logo
    print('Testing case [1/2] started.')
    assert identify_logos_in_images(test_image_urls['logo']) == True, 'Test case [1/2] failed: Expected True, logo should be present.'

    # Test case 2: Image without a logo
    print('Testing case [2/2] started.')
    assert identify_logos_in_images(test_image_urls['no_logo']) == False, 'Test case [2/2] failed: Expected False, no logo should be present.'
    print('Testing finished.')

test_identify_logos_in_images()