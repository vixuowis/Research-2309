# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# function_code --------------------

def transcribe_audio(audio_data):
    """
    Transcribes audio data into text using the pre-trained model 'facebook/wav2vec2-large-960h-lv60-self'.

    Args:
        audio_data (np.array): The audio data to be transcribed.

    Returns:
        transcription (str): The transcribed text.
    """
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-large-960h-lv60-self')

    input_values = processor(audio_data, return_tensors='pt', padding='longest').input_values

    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription

# test_function_code --------------------

def test_transcribe_audio():
    """
    Tests the transcribe_audio function with a sample audio data.
    """
    # Load a sample audio data
    audio_data = load_sample_audio_data()

    # Call the transcribe_audio function
    transcription = transcribe_audio(audio_data)

    # Assert that the transcription is not None
    assert transcription is not None, 'The transcription should not be None.'

    # Assert that the transcription is a string
    assert isinstance(transcription, str), 'The transcription should be a string.'

# call_test_function_code --------------------

test_transcribe_audio()