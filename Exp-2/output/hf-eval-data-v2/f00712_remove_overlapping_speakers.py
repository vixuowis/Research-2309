# function_import --------------------

from transformers import Asteroid
import torch
import soundfile as sf

# function_code --------------------

def remove_overlapping_speakers(input_file_path: str, output_file_path: str):
    """
    This function uses the 'ConvTasNet_Libri2Mix_sepclean_16k' model from Hugging Face Transformers to remove overlapping speakers from an audio recording.

    Args:
        input_file_path (str): The path to the input mixed audio file.
        output_file_path (str): The path to the output separated audio file.

    Returns:
        None. The function writes the separated audio to the output file.
    """
    model = Asteroid('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
    mixed_audio, sample_rate = sf.read(input_file_path)
    mixed_audio_tensor = torch.tensor(mixed_audio)
    separated_audio_tensor = model(mixed_audio_tensor)
    separated_audio = separated_audio_tensor.numpy()
    sf.write(output_file_path, separated_audio, sample_rate)

# test_function_code --------------------

def test_remove_overlapping_speakers():
    """
    This function tests the 'remove_overlapping_speakers' function by using a sample mixed audio file.
    The function asserts that the output file is created and is not empty.
    """
    input_file_path = 'path_to_test_mixed_audio.wav'
    output_file_path = 'path_to_test_separated_audio.wav'
    remove_overlapping_speakers(input_file_path, output_file_path)
    assert os.path.exists(output_file_path), 'Output file not created.'
    assert os.path.getsize(output_file_path) > 0, 'Output file is empty.'

# call_test_function_code --------------------

test_remove_overlapping_speakers()