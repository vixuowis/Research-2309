# function_import --------------------

from transformers import AutoModel
import torch

# function_code --------------------

def get_depth_information(image_path):
    """
    This function uses a pre-trained model from Hugging Face Transformers to estimate the depth of an image.
    The model has been fine-tuned for depth estimation tasks which are useful for robot navigation applications.

    Args:
        image_path (str): The path to the image file.

    Returns:
        depth_information (torch.Tensor): The depth information of the image.
    """
    model = AutoModel.from_pretrained('sayakpaul/glpn-nyu-finetuned-diode-221116-062619')
    preprocessed_image = preprocess_input_image(image_path)
    depth_prediction = model(torch.tensor(preprocessed_image).unsqueeze(0))
    depth_information = extract_depth_info(depth_prediction)
    return depth_information

# test_function_code --------------------

def test_get_depth_information():
    """
    This function tests the get_depth_information function.
    It uses a sample image and checks if the output is a torch.Tensor.
    """
    sample_image_path = 'path_to_sample_image'
    depth_information = get_depth_information(sample_image_path)
    assert isinstance(depth_information, torch.Tensor), 'The output should be a torch.Tensor.'

# call_test_function_code --------------------

test_get_depth_information()