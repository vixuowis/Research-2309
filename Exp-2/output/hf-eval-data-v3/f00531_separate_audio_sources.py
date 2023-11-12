# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import os

# function_code --------------------

def separate_audio_sources(input_audio_file: str, output_file1: str, output_file2: str):
    """
    This function separates the background music and vocal from an audio file using the SepFormer model from SpeechBrain.

    Args:
        input_audio_file (str): The path to the input audio file.
        output_file1 (str): The path to the first output file.
        output_file2 (str): The path to the second output file.

    Returns:
        None
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wsj02mix', savedir='pretrained_models/sepformer-wsj02mix')
    est_sources = model.separate_file(path=input_audio_file)
    torchaudio.save(output_file1, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_file2, est_sources[:, :, 1].detach().cpu(), 8000)

# test_function_code --------------------

def test_separate_audio_sources():
    """
    This function tests the separate_audio_sources function by separating the sources in a test audio file.
    """
    input_audio_file = 'test_audio_file.wav'
    output_file1 = 'test_output1.wav'
    output_file2 = 'test_output2.wav'
    separate_audio_sources(input_audio_file, output_file1, output_file2)
    assert os.path.exists(output_file1), 'Output file 1 does not exist.'
    assert os.path.exists(output_file2), 'Output file 2 does not exist.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_separate_audio_sources()