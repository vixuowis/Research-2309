# requirements_file --------------------

!pip install -U transformers torch soundfile

# function_import --------------------

from transformers import Asteroid
import torch
import soundfile as sf

# function_code --------------------

def separate_speakers(input_audio_path, output_audio_path):
    """
    Separate overlapping speakers in an audio recording using the ConvTasNet_Libri2Mix_sepclean_16k model.

    Args:
        input_audio_path (str): The file path to the mixed audio recording with overlapping speakers.
        output_audio_path (str): The file path where the separated audio will be saved.

    Returns:
        bool: True if the separation was successful, False otherwise.

    Raises:
        IOError: If the input file cannot be opened.
        RuntimeError: If the model fails to process the audio data.
    """
    try:
        model = Asteroid('JorisCos/ConvTasNet_Libri2Mix_sepclean_16k')
        mixed_audio, sample_rate = sf.read(input_audio_path)
        mixed_audio_tensor = torch.tensor(mixed_audio)
        separated_audio_tensor = model(mixed_audio_tensor)
        separated_audio = separated_audio_tensor.numpy()
        sf.write(output_audio_path, separated_audio, sample_rate)
        return True
    except Exception as e:
        print(f'Error occurred: {e}')
        return False

# test_function_code --------------------

def test_separate_speakers():
    print("Testing started.")
    # Assuming we have some test audio files
    input_audio_path = 'test_mixed_audio.wav'
    output_audio_path = 'test_separated_audio.wav'

    print("Testing case [1/1] started.")
    success = separate_speakers(input_audio_path, output_audio_path)
    assert success, f"Test case [1/1] failed: Speaker separation was unsuccessful."
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_speakers()