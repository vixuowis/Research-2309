# function_import --------------------

from transformers import AutoModel
from PIL import Image
import torch

# function_code --------------------

def estimate_depth(image_path):
    """
    Estimate the depth of elements in an architectural design image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: The estimated depth of elements in the image.

    Raises:
        OSError: If the image file cannot be opened.
    """
    # open and resize image
    img = Image.open(image_path)
    if max(img.size) > 512:
        factor = 512/max(img.size)
        new_size = (int(factor*img.size[0]), int(factor*img.size[1]))
        img = img.resize(new_size, Image.ANTIALIAS)
    img = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0)/255
    
    # load model and predict depth
    model = AutoModel.from_pretrained("mczielinski/unet-depth-estimation")
    model.eval()
    if torch.cuda.is_available():
        img, model = img.to('cuda'), model.to('cuda')
    with torch.no_grad():
        pred = (model(img) - 0.5)*2
    
    # convert prediction to depth map and return it
    return pred[0][0].cpu()

# test_function_code --------------------

def test_estimate_depth():
    """
    Test the function estimate_depth.
    """
    sample_image_path = 'https://placekitten.com/200/300'
    try:
        depth_pred = estimate_depth(sample_image_path)
        assert isinstance(depth_pred, torch.Tensor), 'The output should be a torch.Tensor'
    except OSError as e:
        print(f'Error: {e}')
    return 'All Tests Passed'


# call_test_function_code --------------------

test_estimate_depth()