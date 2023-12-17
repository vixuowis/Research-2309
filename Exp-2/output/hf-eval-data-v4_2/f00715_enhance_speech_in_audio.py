# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_speech_in_audio(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhances speech in an audio file using a pre-trained SepFormer model.

    Args:
        input_audio_path (str): The path to the input audio file to be enhanced.
        output_audio_path (str): The path to save the enhanced audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        Exception: If the enhancement process fails.
    """
    # Load the pre-trained SepFormer model
    model = separator.from_hparams(source='speechbrain/sepformer-wham-enhancement', savedir='pretrained_models/sepformer-wham-enhancement')
    
    # Perform speech enhancement
    est_sources = model.separate_file(path=input_audio_path)

    # Save the enhanced audio file
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)

# test_function_code --------------------

def test_enhance_speech_in_audio():
    print("Testing started.")

    # Testing case 1: Valid audio file
    print("Testing case [1/1] started.")
    try:
        enhance_speech_in_audio('valid_input_audio.wav', 'enhanced_audio_output.wav')
        assert os.path.exists('enhanced_audio_output.wav'), "The enhanced audio file was not created."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_speech_in_audio()