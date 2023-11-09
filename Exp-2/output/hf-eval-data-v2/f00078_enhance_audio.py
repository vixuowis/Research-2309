# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhance the audio of noisy recordings using a pretrained Sepformer model.

    Args:
        input_audio_file (str): Path to the input audio file.
        output_audio_file (str): Path to save the enhanced audio file.

    Returns:
        None
    """
    model = separator.from_hparams(source='speechbrain/sepformer-whamr-enhancement', savedir='pretrained_models/sepformer-whamr-enhancement')
    est_sources = model.separate_file(path=input_audio_file)
    torchaudio.save(output_audio_file, est_sources[:, :, 0].detach().cpu(), 8000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Use a sample audio file for testing
    input_audio_file = 'speechbrain/sepformer-whamr-enhancement/example_whamr.wav'
    output_audio_file = 'enhanced_whamr.wav'
    enhance_audio(input_audio_file, output_audio_file)
    # Load the enhanced audio file
    enhanced_audio, _ = torchaudio.load(output_audio_file)
    # Check if the audio file has been enhanced
    assert enhanced_audio.shape[0] == 1, 'The enhanced audio should be mono.'

# call_test_function_code --------------------

test_enhance_audio()