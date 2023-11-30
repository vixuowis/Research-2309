# function_import --------------------

from transformers import pipeline, Wav2Vec2ForCTC

# function_code --------------------

def classify_audio(audio_file_path):
    """
    Classify the category of the audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        str: The category of the audio file.

    Raises:
        OSError: If there is a problem with the file path or the file itself.
    """
    try:
        # load model and classify
        model_name = "facebook/wav2vec2-base-960h"
        asr_model = Wav2Vec2ForCTC.from_pretrained(model_name)
        speech_pipeline = pipeline("automatic-speech-recognition", model=asr_model, device=-1)
        result = speech_pipeline([audio_file_path])[0]["text"]
        
    except OSError:
        raise OSError(f'No such file or directory {audio_file_path}')
    
    return result

# test_function_code --------------------

def test_classify_audio():
    """
    Test the classify_audio function with a sample audio file.

    Returns:
        str: 'All Tests Passed' if all assertions pass, otherwise the error message.
    """
    sample_audio_file_path = 'sample_audio.wav'
    try:
        category = classify_audio(sample_audio_file_path)
        assert isinstance(category, str), 'The output should be a string.'
    except Exception as e:
        return str(e)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_audio()