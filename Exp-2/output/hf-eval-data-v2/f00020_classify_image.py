# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm
import torch

# function_code --------------------

def classify_image(img_url):
    """
    Classify the object within an image using a pretrained MobileNet-v3 model.

    Args:
        img_url (str): The URL of the image to be classified.

    Returns:
        torch.Tensor: The output tensor containing the predicted class probabilities.
    """
    # Load the image
    img = Image.open(urlopen(img_url))

    # Load the pretrained model
    model = timm.create_model('mobilenetv3_large_100.ra_in1k', pretrained=True)
    model = model.eval()

    # Resolve the model data configuration and create the input transforms
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # Apply the transformations to the input image and add a batch dimension
    input_tensor = transforms(img).unsqueeze(0)

    # Pass the transformed image to the model for classification
    output = model(input_tensor)

    return output

# test_function_code --------------------

def test_classify_image():
    """
    Test the classify_image function with a sample image.
    """
    # Define a sample image URL
    img_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'

    # Call the classify_image function
    output = classify_image(img_url)

    # Check the output type and shape
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor'
    assert output.shape == (1, 1000), 'Output shape should be (1, 1000)'

# call_test_function_code --------------------

test_classify_image()