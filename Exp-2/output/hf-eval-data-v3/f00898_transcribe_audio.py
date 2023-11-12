# function_import --------------------

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch

# function_code --------------------

def transcribe_audio(audio_data):
    """
    Transcribe audio data using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_data (np.array): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
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
    Test the transcribe_audio function.
    """
    # Test case: short audio clip
    audio_data = torch.randn(1, 16000)  # 1 second audio clip
    transcription = transcribe_audio(audio_data)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    # Test case: longer audio clip
    audio_data = torch.randn(1, 160000)  # 10 second audio clip
    transcription = transcribe_audio(audio_data)
    assert isinstance(transcription, str), 'The transcription should be a string.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()