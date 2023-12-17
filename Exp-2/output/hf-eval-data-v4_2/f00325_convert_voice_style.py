# requirements_file --------------------

!pip install -U transformers datasets numpy torch soundfile pytest

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch
import numpy as np
from datasets import load_dataset

# function_code --------------------

def convert_voice_style(audio_path:str, embedding_path:str, output_path:str) -> str:
    """
    Convert the voice style of an audio file to a specified style using pre-trained SpeechT5 models.

    Args:
        audio_path (str): The path to the input voiceover audio file.
        embedding_path (str): The path to the numpy file containing the xvector speaker embeddings.
        output_path (str): The path where the converted audio file will be saved.

    Returns:
        str: The path to the converted audio file.

    Raises:
        FileNotFoundError: if the input audio file or embeddings file does not exist.
    """
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    samplerate = 16000
    
    example_speech, _ = sf.read(audio_path)
    speaker_embeddings = np.load(embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    inputs = processor(audio=example_speech, sampling_rate=samplerate, return_tensors='pt')
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)
    sf.write(output_path, speech.numpy(), samplerate)
    
    return output_path

# test_function_code --------------------

def test_convert_voice_style():
    print("Testing started.")
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    dataset = dataset.sort('id')
    test_audio_path = dataset[0]['file']
    test_embedding_path = 'xvector_speaker_embedding.npy'
    output_path = 'test_speech.wav'

    # Test case 1: File exists and embeddings exist
    print("Testing case [1/3] started.")
    result_path = convert_voice_style(test_audio_path, test_embedding_path, output_path)
    assert os.path.isfile(result_path), f"Test case [1/3] failed: output file not found"

    # Test case 2: Audio file does not exist
    print("Testing case [2/3] started.")
    with pytest.raises(FileNotFoundError):
        convert_voice_style('nonexistent_audio.wav', test_embedding_path, output_path)

    # Test case 3: Embeddings file does not exist
    print("Testing case [3/3] started.")
    with pytest.raises(FileNotFoundError):
        convert_voice_style(test_audio_path, 'nonexistent_embeddings.npy', output_path)

    print("Testing finished.")

# call_test_function_line --------------------

test_convert_voice_style()