# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

def identify_spoken_language(audio_file_path):
    """
    Identify the language spoken in an audio file using a pretrained model.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        str: The identified language.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    audio, sample_rate = sf.read(audio_file_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    logits = model(**inputs).logits
    predicted_language_id = logits.argmax(-1).item()
    return predicted_language_id

# test_function_code --------------------

def test_identify_spoken_language():
    """
    Test the identify_spoken_language function.
    """
    audio_file_path = 'test_audio.wav'  # replace with a valid audio file path
    predicted_language_id = identify_spoken_language(audio_file_path)
    assert isinstance(predicted_language_id, int), 'The function should return an integer.'

# call_test_function_code --------------------

test_identify_spoken_language()