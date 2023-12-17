# requirements_file --------------------

!pip install -U timm Pillow

# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm

# function_code --------------------

def classify_image(img_url):
    # Load the image from the provided URL
    img = Image.open(urlopen(img_url))

    # Create the MobileNetV3 model with pretrained weights
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model.eval()  # Set the model to evaluation mode

    # Configure the data and create image transformation
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Preprocess the image
    input_tensor = transforms(img).unsqueeze(0)

    # Classify the image
    output = model(input_tensor)

    # Return the model output
    return output

# test_function_code --------------------

def test_classify_image():
    print('Testing classify_image function.')

    # Test with a sample image URL
    sample_url = 'https://example.com/image.jpg'
    result = classify_image(sample_url)

    # Check if the result is not None (assuming the function returns some output)
    assert result is not None, 'classify_image returned None for a valid URL.'

    # Additional tests can be performed based on the expected output format
    print('classify_image function test passed.')