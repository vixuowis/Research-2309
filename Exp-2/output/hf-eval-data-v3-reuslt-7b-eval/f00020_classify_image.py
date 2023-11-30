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
    
    # Download and open image --------------------
    
    try:
        
        with urlopen(img_url) as url_handle:  # type: BinaryIO
            img = Image.open(url_handle).convert("RGB")  
            
    except URLError:
        raise RuntimeError(f"Could not open image at URL {url}") from None
    
    # Run model --------------------
    
    try:
        
        model = timm.create_model('mobilenetv3_small_100', num_classes=2, in_chans=3)  # type: Module
        model.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/y-bar/face-mask-ml-app/releases/download/v0.0.1/model_mobilenetv3_small_100.pt", map_location="cpu"))
        model.eval()
        
        tensor = transforms.ToTensor()(img).unsqueeze_(0)  # type: Tensor
            
    except Exception as e:
        raise RuntimeError("An error occured when running the model.") from e
    
    with torch.no_grad():
        
        output = model(tensor)  # type: Tensor
        _, prediction = torch.max(output, dim=1)  # type: Tuple[Tensor, Tensor]
            
    return int(prediction)

# test_function_code --------------------

def test_classify_image():
    """Test the classify_image function."""
    assert isinstance(classify_image('https://placekitten.com/200/300'), int)
    assert isinstance(classify_image('https://placekitten.com/200/301'), int)
    assert isinstance(classify_image('https://placekitten.com/200/302'), int)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_image()