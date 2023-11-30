# function_import --------------------

from urllib.request import urlopen
from PIL import Image
import timm
import torch

# function_code --------------------

def classify_image(img_url: str) -> int:
    """
    Classify an image using a pretrained MobileNet-v3 model.

    Args:
        img_url (str): The URL of the image to classify.

    Returns:
        int: The predicted class of the image.

    Raises:
        URLError: If the image cannot be opened from the provided URL.
        RuntimeError: If there is a problem running the model.
    """
    # Download and load the image.
    img_bytes = urlopen(img_url).read()
    input_image = Image.open(io.BytesIO(img_bytes))
    
    # Resize it to 256 x 256, as expected by MobileNet-v3.
    resized_image = input_image.resize((256, 256))

    # Convert the image to tensors and run through the model.
    tensor_transform = timm.data.TensorTransform()
    processed_tensor = tensor_transform(resized_image)
    batched_tensors = torch.unsqueeze(processed_tensor, dim=0)
    
    model = timm.create_model("mobilenetv3_small_100", pretrained=True)
    logits = model(batched_tensors)
    predicted_class = torch.argmax(logits).item()

    return predicted_class

# function_test --------------------

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    assert isinstance(classify_image('https://placekitten.com/200/300'), int)
    assert isinstance(classify_image('https://placekitten.com/200/301'), int)
    assert isinstance(classify_image('https://placekitten.com/200/302'), int)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()