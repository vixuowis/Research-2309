# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio_quality(input_path, output_path):
    '''
    Enhances the audio quality of a given file by removing background noise.

    Parameters:
        input_path (str): Path to the input audio file that needs speech enhancement.
        output_path (str): Path to save the enhanced audio file.

    Returns:
        None: The enhanced audio is saved to the output_path.
    '''
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_path)
    torchaudio.save(output_path, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio_quality():
    print('Testing started.')
    input_path = 'test_podcast.wav'  # This should be a path to a test audio file with noise
    output_path = 'enhanced_test_podcast.wav'

    # Call the enhance audio quality function
    enhance_audio_quality(input_path, output_path)

    # Verify the output file exists
    assert os.path.exists(output_path), 'Test failed: output file does not exist.'
    print('Test passed: Enhanced audio file created.')
    print('Testing finished.')

# Run the test function
test_enhance_audio_quality()