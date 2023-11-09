# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_speech(input_audio_path: str, output_audio_path: str = 'enhanced_audio.wav') -> None:
    """
    Enhances the speech in an audio file using the SepFormer model.

    Args:
        input_audio_path (str): The path to the input audio file.
        output_audio_path (str, optional): The path to save the enhanced audio file. Defaults to 'enhanced_audio.wav'.

    Returns:
        None
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham-enhancement', savedir='pretrained_models/sepformer-wham-enhancement')
    est_sources = model.separate_file(path=input_audio_path)
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)

# test_function_code --------------------

def test_enhance_speech():
    """
    Tests the enhance_speech function.
    """
    # Use a sample audio file for testing
    input_audio_path = 'speechbrain/sepformer-wham-enhancement/example_wham.wav'
    output_audio_path = 'enhanced_wham.wav'
    enhance_speech(input_audio_path, output_audio_path)
    # Load the enhanced audio file
    enhanced_audio, _ = torchaudio.load(output_audio_path)
    # Check if the audio file was enhanced (not strictly accurate)
    assert enhanced_audio.shape[0] > 0

# call_test_function_code --------------------

test_enhance_speech()