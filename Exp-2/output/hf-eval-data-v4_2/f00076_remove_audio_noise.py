# requirements_file --------------------

!pip install -U torch torchaudio speechbrain

# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# function_code --------------------

def remove_audio_noise(audio_file_path, enhanced_file_path='enhanced.wav'):
    """Remove noise from an audio file using a pre-trained speech enhancement model.

    Args:
        audio_file_path (str): The path to the noisy audio file.
        enhanced_file_path (str): The path where the enhanced audio will be saved. Defaults to 'enhanced.wav'.

    Returns:
        str: The path to the enhanced audio file.

    Raises:
        FileNotFoundError: If the audio_file_path does not exist.
        RuntimeError: If the enhancement process fails.
    """
    # Load the pre-trained speech enhancement model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    # Load the noisy audio file
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f'The audio file {audio_file_path} was not found.')

    noisy = enhance_model.load_audio(audio_file_path).unsqueeze(0)
    # Enhance the audio
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Save the enhanced audio
    torchaudio.save(enhanced_file_path, enhanced.cpu(), 16000)

    return enhanced_file_path

# test_function_code --------------------

def test_remove_audio_noise():
    print("Testing started.")
    # Assuming we have a sample noisy audio file for testing
    sample_noisy_audio = 'sample_noisy.wav'
    enhanced_audio_file = 'sample_enhanced.wav'

    # Test case 1: Check if the function returns the correct path for the enhanced file
    print("Testing case [1/3] started.")
    result_path = remove_audio_noise(sample_noisy_audio, enhanced_audio_file)
    assert result_path == enhanced_audio_file, f"Test case [1/3] failed: Expected {enhanced_audio_file}, got {result_path}."

    # Test case 2: Check if the enhanced file is indeed created
    print("Testing case [2/3] started.")
    assert os.path.isfile(enhanced_audio_file), f"Test case [2/3] failed: The file {enhanced_audio_file} was not created."

    # Test case 3: Check if an error is raised when providing a non-existent file path
    print("Testing case [3/3] started.")
    try:
        remove_audio_noise('non_existent_file.wav')
        assert False, "Test case [3/3] failed: FileNotFoundError was expected but not raised."
    except FileNotFoundError:
        pass  # Expected exception

    print("Testing finished.")

# call_test_function_line --------------------

test_remove_audio_noise()