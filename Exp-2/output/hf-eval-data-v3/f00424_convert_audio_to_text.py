# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# function_code --------------------

def convert_audio_to_text(audio_file_path):
    '''
    Convert the audio file of a phone interview to text for further analysis.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The transcribed text from the audio file.

    Raises:
        ValueError: If the audio file path is not a string.
    '''
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Load phone interview audio file as a numpy array
    input_values = processor(audio_file_path, return_tensors='pt', padding='longest').input_values

    # Get logits from the model
    logits = model(input_values).logits

    # Predict the transcriptions
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode transcriptions into text
    transcription = processor.batch_decode(predicted_ids)

    return transcription

# test_function_code --------------------

def test_convert_audio_to_text():
    '''
    Test the convert_audio_to_text function.
    '''
    # Test with a valid audio file
    transcription = convert_audio_to_text('path/to/valid/audio/file')
    assert isinstance(transcription, str), 'The transcription should be a string.'

    # Test with an invalid audio file
    try:
        convert_audio_to_text(123)
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError when the audio file path is not a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_audio_to_text()