# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio_quality(input_audio_path, output_audio_path):
    """
    Enhances the quality of the input audio using a pre-trained model.

    Args:
        input_audio_path (str): The file path to the low-quality audio to enhance.
        output_audio_path (str): The file path where the enhanced audio will be saved.

    Returns:
        str: The file path to the enhanced audio file.
    """
    # Load the pre-trained sepformer model for audio enhancement
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement',
                                   savedir='pretrained_models/sepformer-wham16k-enhancement')
    # Perform speech enhancement on the low-quality audio file
    est_sources = model.separate_file(path=input_audio_path)
    # Save the enhanced audio file
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)
    
    return output_audio_path

# test_function_code --------------------

def test_enhance_audio_quality():
    print("Testing started.")
    input_audio_sample = "path_to_sample_low_quality_audio.wav"  # Low-quality sample file
    output_audio_path = "output_enhanced_audio.wav"  # Path to save enhanced audio

    # Test case 1: Check if the enhanced audio file is saved correctly
    print("Testing case [1/1] started.")
    enhanced_audio_path = enhance_audio_quality(input_audio_sample, output_audio_path)
    assert os.path.exists(enhanced_audio_path), f"Test case [1/1] failed: Enhanced audio file was not saved."
    
    print("Testing finished.")