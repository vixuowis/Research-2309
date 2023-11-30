# function_import --------------------

from pyannote.audio import Model

# function_code --------------------

def load_voice_activity_detection_model():
    """
    Load the pre-trained voice activity detection model from Hugging Face Transformers.

    Returns:
        model: The pre-trained voice activity detection model.
    """
    return Model.from_pretrained(
        "pyannote/segmentation",
        map_location=None,
        strict=False,
        use_auth_token=str(os.environ["HF_AUTH_TOKEN"]),
    )


# test_function_code --------------------

def test_load_voice_activity_detection_model():
    """
    Test the function load_voice_activity_detection_model.
    """
    model = load_voice_activity_detection_model()
    assert model is not None, 'Model loading failed.'
    print('All Tests Passed')


# call_test_function_code --------------------

test_load_voice_activity_detection_model()