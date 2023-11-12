# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_file: str, output_file: str) -> None:
    """
    Enhance the audio quality of an input file by reducing noise using a pre-trained model.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input file does not exist.
        Exception: If there is an error in enhancing the audio.
    """
    try:
        # Load the pre-trained model
        model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
        # Separate speech from the noise in the target audio file
        est_sources = model.separate_file(path=input_file)
        # Save the enhanced audio file to disk
        torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 16000)
    except FileNotFoundError as fnf_error:
        print(f'File not found: {fnf_error}')
        raise
    except Exception as e:
        print(f'Error in enhancing audio: {e}')
        raise

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function with a sample audio file.

    Returns:
        str: 'All Tests Passed' if all assertions pass, else an error message.
    """
    # Test with a sample audio file
    input_file = 'sample_audio.wav'
    output_file = 'enhanced_audio.wav'
    try:
        enhance_audio(input_file, output_file)
        # Check if the output file was created
        assert os.path.exists(output_file), 'Output file not created'
        # Check if the output file is not empty
        assert os.path.getsize(output_file) > 0, 'Output file is empty'
        return 'All Tests Passed'
    except AssertionError as error:
        return str(error)

# call_test_function_code --------------------

print(test_enhance_audio())