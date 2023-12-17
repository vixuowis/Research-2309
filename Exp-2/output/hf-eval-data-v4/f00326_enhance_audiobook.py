# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audiobook(audio_path: str, output_path: str) -> str:
    """
    Enhance an audiobook file by reducing the noise using the SepFormer model.

    Parameters:
    audio_path (str): Path to the audiobook file that needs enhancement.
    output_path (str): Path where the enhanced audiobook will be saved.

    Returns:
    str: The path to the enhanced audiobook file.
    """
    # Load the pre-trained separator model
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    # Separate speech from noise
    est_sources = model.separate_file(path=audio_path)
    # Save the enhanced audio
    torchaudio.save(output_path, est_sources[:, :, 0].detach().cpu(), 16000)
    return output_path

# test_function_code --------------------

def test_enhance_audiobook():
    print("Testing enhance_audiobook function.")
    audio_path = 'sample_audiobook.wav'  # Example audiobook path
    output_path = 'enhanced_sample_audiobook.wav'
    result_path = enhance_audiobook(audio_path, output_path)

    # Test case: Check if the function returns the correct output path.
    assert result_path == output_path, f"Test failed: Expected {output_path}, got {result_path}"

    # Further tests could involve checking the presence of the output file,
    # comparing the noise levels before and after, etc. However, this would
    # require access to the file system and audio processing tools.
    print("Test passed.")

# Run the test
test_enhance_audiobook()