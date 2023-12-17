# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audiobook(path_to_audiobook):
    """
    Enhance the audio quality of an audiobook by separating speech from noise using a pretrained model.

    Args:
        path_to_audiobook (str): Path to the audiobook file that needs enhancement.

    Returns:
        Tensor: The enhanced audio tensor.

    Raises:
        FileNotFoundError: If the audiobook file does not exist.
        RuntimeError: If the separation model encounters an error during processing.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    try:
        est_sources = model.separate_file(path=path_to_audiobook)
        enhanced_audio = est_sources[:, :, 0]
        return enhanced_audio.detach().cpu()
    except Exception as e:
        raise RuntimeError(f'Error while enhancing audio: {e}')

# test_function_code --------------------

def test_enhance_audiobook():
    print("Testing started.")
    test_audio_path = 'test_audiobook.wav'  # Replace with a path to a test audiobook file

    # Test case 1: Valid audio file
    print("Testing case [1/3] started.")
    enhanced_audio = enhance_audiobook(test_audio_path)
    assert enhanced_audio is not None, f"Test case [1/3] failed: Expected enhanced audio, got {enhanced_audio}"

    # Test case 2: Non-existing audio file
    print("Testing case [2/3] started.")
    try:
        enhance_audiobook('non_existing_file.wav')
        assert False, "Test case [2/3] failed: Expected FileNotFoundError, did not occur."
    except FileNotFoundError:
        assert True

    # Test case 3: Invalid audio file format
    print("Testing case [3/3] started.")
    try:
        enhance_audiobook('invalid_audio_file.txt')
        assert False, "Test case [3/3] failed: Expected RuntimeError, did not occur."
    except RuntimeError:
        assert True
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_audiobook()