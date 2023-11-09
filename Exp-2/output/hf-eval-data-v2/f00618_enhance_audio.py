# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_file: str, output_file: str = 'enhanced_audio.wav'):
    """
    Enhance the quality of an audio file by removing background noise using a pre-trained model from SpeechBrain.

    Args:
        input_file (str): Path to the input audio file that needs speech enhancement.
        output_file (str, optional): Path where the enhanced audio will be saved. Defaults to 'enhanced_audio.wav'.

    Returns:
        None. The enhanced audio is saved to the specified output file.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_file)
    torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function with a sample audio file.
    """
    input_file = 'example_podcast.wav'
    output_file = 'enhanced_podcast.wav'
    enhance_audio(input_file, output_file)
    assert os.path.exists(output_file), 'Enhanced audio file not found.'

# call_test_function_code --------------------

test_enhance_audio()