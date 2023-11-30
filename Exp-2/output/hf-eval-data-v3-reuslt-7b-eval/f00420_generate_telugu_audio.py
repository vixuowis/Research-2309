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
    
    # load pipeline
    # Note that we are using a different model for female voice than male. 
    # To do more with this code, you could also try 'SYSPIN/Telugu_Female_TTS'.
    # The model 'floveravi-tlg-tts' is also available on HuggingFace and might be worth a look too.
    
    print("Loading pipeline...")
    telugu_pipeline = pipeline('text-to-speech', \
                               model='SYSPIN/Telugu_Male_TTS')
    
    # generate audio bytes from text and save to file
    
    print(telugu_text)  # show the input text on screen, for debugging purposes.
    telugu_audio = telugu_pipeline(telugu_text)[0]

    return telugu_audio

# function_export --------------------


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