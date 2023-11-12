# function_import --------------------

from huggingface_hub import hf_hub_download

# function_code --------------------

def download_and_load_model(repo_id: str, filename: str) -> str:
    """
    Download and load a model from Hugging Face Model Hub.

    Args:
        repo_id (str): The repository ID of the model to download.
        filename (str): The filename of the model file.

    Returns:
        str: The local path of the downloaded model.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    return model_path

# test_function_code --------------------

def test_download_and_load_model():
    """
    Test the function download_and_load_model.
    """
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    filename = 'model'
    model_path = download_and_load_model(repo_id, filename)
    assert isinstance(model_path, str), 'The returned model path should be a string.'
    assert model_path.endswith(filename), 'The model path should end with the filename.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_download_and_load_model()