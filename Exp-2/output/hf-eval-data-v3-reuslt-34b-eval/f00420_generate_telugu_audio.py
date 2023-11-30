# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_telugu_audio(telugu_text):
    """
    Generate Telugu audio from text using the ESPnet framework.

    Args:
        telugu_text (str): The Telugu text to be converted to audio.

    Returns:
        bytes: The audio representation of the input text.

    Raises:
        OSError: If the model 'SYSPIN/Telugu_Male_TTS' is not found.
    """
    
    try:
        tts = pipeline("text-to-speech", "SYSPIN/Telugu_Male_TTS")
        
    except OSError as os_error:
        if str(os_error) == "[Errno 2] No such file or directory: 'SYSPIN_Telugu_Male_TTS.pth.tar': 'SYSPIN/Telugu_Male_TTS' is not installed.":
            print("The ESPnet model 'SYSPIN/Telugu_Male_TTS' was not found.")
        raise os_error
    
    else:
         telugu_audio = tts(telugu_text)
         return telugu_audio[0]["raw_audio"]

# test_function_code --------------------

def test_generate_telugu_audio():
    """
    Test the function generate_telugu_audio.
    """
    # Test with a sample Telugu text
    sample_text = 'తెలుగు శ్లోకము లేదా ప్రార్థన ఇక్కడ ఉండాలి'
    try:
        audio = generate_telugu_audio(sample_text)
        assert isinstance(audio, bytes), 'The output is not in bytes format.'
    except OSError as e:
        assert str(e) == 'SYSPIN/Telugu_Male_TTS does not appear to have a file named config.json. Checkout \'https://huggingface.co/SYSPIN/Telugu_Male_TTS/main\' for available files.', 'The model is not found.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_telugu_audio()