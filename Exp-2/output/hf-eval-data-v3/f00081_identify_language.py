# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import torch
import soundfile as sf

# function_code --------------------

def identify_language(audio_file_path: str) -> str:
    """
    Identify the language spoken in an audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The identified language.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        RuntimeError: If the model or processor cannot be loaded.
    """
    # Load the model and processor
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    # Load the audio file
    speech, _ = sf.read(audio_file_path)

    # Process the audio file and make a prediction
    inputs = processor(speech, return_tensors='pt', padding=True)
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits
    predicted_id = torch.argmax(logits, dim=-1)

    # Return the identified language
    return model.config.id2label[predicted_id.item()]

# test_function_code --------------------

def test_identify_language():
    """Test the identify_language function."""
    # Test with a valid audio file
    assert identify_language('valid_audio_file.wav') == 'English'
    # Test with an invalid audio file
    try:
        identify_language('invalid_audio_file.wav')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')
    # Test with a non-audio file
    try:
        identify_language('non_audio_file.txt')
    except RuntimeError:
        pass
    else:
        raise AssertionError('Expected a RuntimeError.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_language()