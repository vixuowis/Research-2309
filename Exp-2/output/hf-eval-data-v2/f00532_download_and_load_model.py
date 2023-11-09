# function_import --------------------

from huggingface_hub import hf_hub_download

# function_code --------------------

def download_and_load_model(repo_id: str):
    """
    Download and load the pre-trained ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face Transformers.

    Args:
        repo_id (str): The repository ID of the model on Hugging Face Transformers.

    Returns:
        The path to the downloaded model.
    """
    model_path = hf_hub_download(repo_id=repo_id)
    return model_path

# test_function_code --------------------

def test_download_and_load_model():
    """
    Test the download_and_load_model function.

    Raises:
        AssertionError: If the function does not return a valid model path.
    """
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    model_path = download_and_load_model(repo_id)
    assert isinstance(model_path, str) and len(model_path) > 0, 'The function should return a valid model path.'

# call_test_function_code --------------------

test_download_and_load_model()