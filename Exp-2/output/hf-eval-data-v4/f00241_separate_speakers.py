# requirements_file --------------------

!pip install -U torch asteroid pydub 

# function_import --------------------

import torch
from asteroid.models import ConvTasNet
from pydub import AudioSegment


# function_code --------------------

def separate_speakers(mixed_audio_path):
    """
    Use a pretrained ConvTasNet model to separate individual speaker voices from a mixed audio with multiple speakers.
    
    Args:
        mixed_audio_path (str): Path to the mixed audio file.
    
    Returns:
        List[AudioSegment]: A list of AudioSegments for each separated voice track.
    """
    # Load the pretrained ConvTasNet model
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    
    # Load the mixed audio
    mixed_audio = AudioSegment.from_file(mixed_audio_path).set_frame_rate(8000).set_channels(1)
    
    # Prepare input tensor
    mixed_signal = torch.tensor(mixed_audio.get_array_of_samples(), dtype=torch.float).unsqueeze(0)
    
    # Perform separation
    separated_sources = model.separate(mixed_signal)
    
    # Convert separated sources to AudioSegments and return
    separated_audio_segments = [AudioSegment(separated_source.numpy(), frame_rate=8000, sample_width=2, channels=1)
                                for separated_source in separated_sources]
    
    return separated_audio_segments


# test_function_code --------------------

def test_separate_speakers():
    print("Testing started.")
    # Assuming there is a test mixed audio file from a dataset or a generated one
    test_audio_path = "test_mixed_audio.wav"
    
    # Test case 1: Separate speakers from mixed audio
    print("Testing case [1/1] started.")
    separated_speakers = separate_speakers(test_audio_path)

    # Assume there's a way to assess the length of separated sources; it should match the number of sources (3)
    assert len(separated_speakers) == 3, f"Test case [1/1] failed: Expected 3 separated sources, got {len(separated_speakers)} instead."
    
    # Add additional checks if ground truth data is available for comparison
    print("Testing finished.")

# Run the test function
test_separate_speakers()
