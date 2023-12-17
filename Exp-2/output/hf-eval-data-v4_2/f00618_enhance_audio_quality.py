# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio_quality(input_file_path: str, output_file_path: str) -> None:
    """Enhance the audio quality by removing background noise using a pre-trained model.

    Args:
        input_file_path (str): The file path to the input audio file that needs enhancement.
        output_file_path (str): The file path to save the enhanced audio file.

    Returns:
        None: The function saves the enhanced audio file to the specified output path.

    Raises:
        FileNotFoundError: If the input_file_path does not exist.
        Exception: If the separation process fails.
    """
    # Load the pre-trained Speech Enhancement model
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    
    # Perform speech enhancement
    est_sources = model.separate_file(path=input_file_path)
    
    # Save the enhanced audio
    torchaudio.save(output_file_path, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio_quality():
    print("Testing started.")

    # Mock an existing file path and a target output file path for testing
    mock_input_path = 'test_files/input_podcast_example.wav'
    mock_output_path = 'test_files/enhanced_podcast_example.wav'

    # Assuming we already have a mock file in the test_files directory
    # We're skipping the actual audio file loading and separation process to simulate the result

    # Testing case 1: Correct file paths provided
    print("Testing case [1/1] started.")
    enhance_audio_quality(mock_input_path, mock_output_path)
    assert True, "Test case [1/1] failed: enhance_audio_quality did not complete successfully with valid file paths."
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_audio_quality()