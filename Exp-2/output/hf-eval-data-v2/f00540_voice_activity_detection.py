# function_import --------------------

from pyannote.audio import Model

# function_code --------------------

def voice_activity_detection():
    """
    This function is used to perform voice activity detection in audio recordings.
    It uses a pre-trained model from the Hugging Face Transformers library.
    The model is 'popcornell/pyannote-segmentation-chime6-mixer6' which is specifically designed for this task.
    
    Returns:
    model: A pre-trained model for voice activity detection.
    """
    model = Model.from_pretrained('popcornell/pyannote-segmentation-chime6-mixer6')
    return model

# test_function_code --------------------

def test_voice_activity_detection():
    """
    This function is used to test the voice_activity_detection function.
    It asserts if the returned model is not None.
    """
    model = voice_activity_detection()
    assert model is not None, 'Model not loaded.'

# call_test_function_code --------------------

test_voice_activity_detection()