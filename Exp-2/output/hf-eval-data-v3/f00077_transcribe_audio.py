# function_import --------------------

from transformers import Wav2Vec2Model
from datasets import load_dataset
import soundfile as sf
import torch

# function_code --------------------

def transcribe_audio(audio_paths):
    """
    Transcribe audio files into Chinese text using Hugging Face's Wav2Vec2Model.

    Args:
        audio_paths (list): A list of paths to audio files.

    Returns:
        list: A list of transcriptions for each audio file.
    """
    model = Wav2Vec2Model.from_pretrained('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    transcriptions = []
    for path in audio_paths:
        speech, _ = sf.read(path)
        input_values = model.processor(speech, return_tensors='pt').input_values
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = model.processor.decode(predicted_ids[0])
        transcriptions.append(transcription)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function.
    """
    # Load a sample audio file from the 'common_voice' dataset
    dataset = load_dataset('common_voice', 'zh-CN', split='train[:1]')
    audio_path = dataset[0]['path']
    transcription = transcribe_audio([audio_path])
    assert isinstance(transcription, list), 'The result should be a list.'
    assert len(transcription) == 1, 'The list should contain one transcription.'
    assert isinstance(transcription[0], str), 'The transcription should be a string.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_transcribe_audio()