# requirements_file --------------------

!pip install -U transformers==4.27.0.dev0 torch==1.13.1 datasets==2.9.0 tokenizers==0.13.2 soundfile

# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

def identify_spoken_language(audio_file_path):
    """
    Identify the language being spoken in an audio file using a pretrained model.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        int: The ID of the identified language.

    Raises:
        FileNotFoundError: If the audio file does not exist at the given path.
        RuntimeError: If there is an error processing the audio data.
    """
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    try:
        audio, sample_rate = sf.read(audio_file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f'The audio file was not found at the path: {audio_file_path}')

    try:
        inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        logits = model(**inputs).logits
        predicted_language_id = logits.argmax(-1).item()
        return predicted_language_id
    except Exception as e:
        raise RuntimeError(f'Error processing the audio data: {e}')

# test_function_code --------------------

def test_identify_spoken_language():
    print("Testing started.")

    # Test case 1: Valid audio file
    print("Testing case [1/2] started.")
    predicted_id = identify_spoken_language('path/to/valid/audio.wav')
    assert isinstance(predicted_id, int), f"Test case [1/2] failed: Expected an integer language ID, got {type(predicted_id).__name__} instead."

    # Test case 2: Non-existent audio file
    print("Testing case [2/2] started.")
    try:
        identify_spoken_language('path/to/nonexistent/audio.wav')
        assert False, "Test case [2/2] failed: FileNotFoundError was expected but not raised."
    except FileNotFoundError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_identify_spoken_language()