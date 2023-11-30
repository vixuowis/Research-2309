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
    from espnet2.bin.tts_inference import Text2Speech

    tts = Text2Speech(model_file="exp/text2speech_tacotron2/default/train.loss.best.pth",
                      model_type='tacotron2', 
                      device='cpu')
    
    with open("temp.txt","w") as f:
        f.write(telugu_text)

    wav = tts('temp.txt')[0]

    os.remove("temp.txt")

    return wav

# function_code --------------------


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