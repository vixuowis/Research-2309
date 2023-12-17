# requirements_file --------------------

!pip install -U speechbrain

# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# function_code --------------------

def enhance_coworker_call(audio_path):
    """
    Enhances the audio quality of a noisy call recording using a pre-trained MetricGAN model.
    
    Args:
    - audio_path (str): The file path to the noisy audio recording.
    
    Returns:
    - enhanced_waveform (Tensor): The enhanced audio waveform.
    - sample_rate (int): The sample rate of the audio recording.
    """
    
    # Initialize the pre-trained enhancement model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    
    # Load the noisy audio and unsqueeze to add a batch dimension
    noisy_waveform = enhance_model.load_audio(audio_path).unsqueeze(0)
    
    # Perform enhancement on the noisy audio
    lengths = torch.tensor([1.0])  # Indicates the relative length of the audio, 1.0 for full audio
    enhanced_waveform = enhance_model.enhance_batch(noisy_waveform, lengths=lengths)
    
    # Get the sample rate for saving the enhanced audio
    sample_rate = enhance_model.hparams.sample_rate
    
    return enhanced_waveform.squeeze(0), sample_rate

# test_function_code --------------------

def test_enhance_coworker_call():
    print("Testing started.")
    
    # Assuming 'coworker_call_noisy.wav' is a noisy sample that exists in the current directory
    audio_path = 'coworker_call_noisy.wav'
    
    # Test case 1: The function returns an audio waveform tensor and a sample rate integer
    print("Testing case [1/1] started.")
    enhanced_waveform, sample_rate = enhance_coworker_call(audio_path)
    assert isinstance(enhanced_waveform, torch.Tensor), f"Test case [1/1] failed: The result is not a tensor."
    assert isinstance(sample_rate, int), f"Test case [1/1] failed: The sample rate is not an integer."
    assert enhanced_waveform.ndim == 1, f"Test case [1/1] failed: The result tensor is not 1D."
    
    # Saving the enhanced audio to a file for manual inspection if needed
    torchaudio.save('coworker_call_enhanced.wav', enhanced_waveform.unsqueeze(0), sample_rate)
    
    print("Testing finished.")

# Run the test function
test_enhance_coworker_call()