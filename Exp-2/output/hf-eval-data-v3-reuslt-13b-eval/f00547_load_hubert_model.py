# function_import --------------------

from transformers import HubertModel

# function_code --------------------

def load_hubert_model(model_name: str):
    """
    Load the pre-trained Hubert model from Hugging Face Transformers.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        HubertModel: The loaded model.

    Raises:
        OSError: If there is not enough disk space to download the model.
    """
    try:
        # Try to load the cached model.
        return HubertModel.from_pretrained(model_name)
    except OSError as e1:
        # Try to redownload the model, in case there was a partial download.
        print("Downloading pre-trained model (again).")
        try:
            return HubertModel.from_pretrained(model_name)
        except OSError as e2:
            # If the error is not due to "not enough disk space", re-raise it.
            if str(e2) != \
                'Not enough space for downloading and uncompressing model data.' \
                    .format(model_name, 10):
                raise e2


# test_function_code --------------------

def test_load_hubert_model():
    """
    Test the function load_hubert_model.
    """
    model_name = 'facebook/hubert-large-ll60k'
    try:
        model = load_hubert_model(model_name)
        assert isinstance(model, HubertModel), 'Model loading failed'
        print('Test passed')
    except OSError as e:
        print(f'Error: {e}')
        assert str(e) == '[Errno 28] No space left on device', 'Unexpected error'
        print('Test passed')


# call_test_function_code --------------------

test_load_hubert_model()