# requirements_file --------------------

!pip install -U torch torchaudio speechbrain

# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# function_code --------------------

def remove_noise_from_audio(input_audio_path, output_audio_path):
    # Import the necessary libraries and modules from SpeechBrain and Torchaudio
    # Load the pre-trained speech enhancement model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    # Load and process the noisy input audio file
    noisy_audio = enhance_model.load_audio(input_audio_path).unsqueeze(0)
    # Enhance the audio using the pre-trained model
    enhanced_audio = enhance_model.enhance_batch(noisy_audio, lengths=torch.tensor([1.]))
    # Save the enhanced audio to the output path
    torchaudio.save(output_audio_path, enhanced_audio.cpu(), 16000)
    return output_audio_path

# test_function_code --------------------

def test_remove_noise_from_audio():
    print("Testing remove_noise_from_audio function.")
    # Path to a sample noisy audio file for testing
    test_input_audio_path = 'path/to/test/noisy_audio_file.wav'
    test_output_audio_path = 'path/to/test/enhanced_audio.wav'
    # Call the function with the test paths
    remove_noise_from_audio(test_input_audio_path, test_output_audio_path)
    # Load the enhanced audio file and check if it exists
    assert os.path.exists(test_output_audio_path), f"Test failed: Enhanced audio file was not created at {test_output_audio_path}"
    # Additional logic to compare the enhanced audio with a ground truth could be added here
    print("Test passed: Enhanced audio file was successfully created.")

# Execute the test function
if __name__ == '__main__':
    test_remove_noise_from_audio()