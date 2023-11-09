# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio_quality(input_audio_path: str, output_audio_path: str = 'enhanced_audio.wav') -> str:
    """
    Enhances the audio quality of a given audio file using a pre-trained model from SpeechBrain.

    Args:
        input_audio_path (str): Path to the input low-quality audio file.
        output_audio_path (str, optional): Path where the enhanced audio file will be saved. Defaults to 'enhanced_audio.wav'.

    Returns:
        str: Path to the enhanced audio file.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_audio_path)
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)
    return output_audio_path

# test_function_code --------------------

def test_enhance_audio_quality():
    """
    Tests the enhance_audio_quality function by enhancing a sample low-quality audio file.
    """
    input_audio_path = 'path_to_test_low_quality_audio.wav'
    output_audio_path = 'test_enhanced_audio.wav'
    enhanced_audio_path = enhance_audio_quality(input_audio_path, output_audio_path)
    assert enhanced_audio_path == output_audio_path
    assert os.path.exists(enhanced_audio_path)

# call_test_function_code --------------------

test_enhance_audio_quality()