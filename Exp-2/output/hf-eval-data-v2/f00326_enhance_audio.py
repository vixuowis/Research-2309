# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_file: str, output_file: str = 'enhanced_audiobook.wav'):
    """
    Enhances the audio quality of an audiobook by separating speech from noise.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced_audiobook.wav'.

    Returns:
        None
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_file)
    torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Tests the enhance_audio function by enhancing a sample audio file.
    """
    input_file = 'test_audiobook.wav'
    output_file = 'test_enhanced_audiobook.wav'
    enhance_audio(input_file, output_file)
    enhanced_audio, _ = torchaudio.load(output_file)
    assert enhanced_audio is not None

# call_test_function_code --------------------

test_enhance_audio()