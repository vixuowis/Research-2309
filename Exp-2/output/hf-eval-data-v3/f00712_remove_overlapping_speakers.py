# function_import --------------------

from asteroid import ConvTasNet
import torch
import soundfile as sf

# function_code --------------------

def remove_overlapping_speakers(input_file_path: str, output_file_path: str):
    '''
    This function uses the ConvTasNet_Libri2Mix_sepclean_16k model from Hugging Face Transformers to remove overlapping speakers from an audio recording.

    Args:
        input_file_path (str): The path to the input mixed audio file.
        output_file_path (str): The path to the output separated audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input file does not exist.
        RuntimeError: If there is an error during the processing of the audio file.
    '''
    model = ConvTasNet.from_pretrained('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
    mixed_audio, sample_rate = sf.read(input_file_path)
    mixed_audio_tensor = torch.tensor(mixed_audio)
    separated_audio_tensor = model(mixed_audio_tensor)
    separated_audio = separated_audio_tensor.numpy()
    sf.write(output_file_path, separated_audio, sample_rate)

# test_function_code --------------------

def test_remove_overlapping_speakers():
    '''
    This function tests the remove_overlapping_speakers function.
    '''
    # Test case 1: Normal case
    remove_overlapping_speakers('test_data/mixed_audio.wav', 'test_data/separated_audio.wav')
    assert os.path.exists('test_data/separated_audio.wav')

    # Test case 2: Input file does not exist
    try:
        remove_overlapping_speakers('test_data/non_existent_file.wav', 'test_data/separated_audio.wav')
    except FileNotFoundError:
        pass

    # Test case 3: Output file cannot be written
    try:
        remove_overlapping_speakers('test_data/mixed_audio.wav', '/non_writable_directory/separated_audio.wav')
    except PermissionError:
        pass

    return 'All Tests Passed'

# call_test_function_code --------------------

test_remove_overlapping_speakers()