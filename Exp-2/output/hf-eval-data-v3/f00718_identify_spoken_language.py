# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

def identify_spoken_language(audio_file_path):
    """
    Identify the language spoken in an audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The identified language.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    audio, sample_rate = sf.read(audio_file_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors='pt')
    logits = model(**inputs).logits
    predicted_language_id = logits.argmax(-1).item()
    return predicted_language_id

# test_function_code --------------------

def test_identify_spoken_language():
    assert identify_spoken_language('test_audio_english.wav') == 'English'
    assert identify_spoken_language('test_audio_french.wav') == 'French'
    assert identify_spoken_language('test_audio_spanish.wav') == 'Spanish'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_spoken_language()