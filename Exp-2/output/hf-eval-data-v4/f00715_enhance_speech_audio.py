# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_speech_audio(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhance the speech in an audio file using a pre-trained SepFormer model.

    Args:
        input_audio_path (str): The path to the input audio file to be processed.
        output_audio_path (str): The path where the enhanced audio file will be saved.
    """
    # Load the pre-trained SepFormer model
    model = separator.from_hparams(source='speechbrain/sepformer-wham-enhancement', savedir='pretrained_models/sepformer-wham-enhancement')
    # Process the input audio file and separate the sources
    est_sources = model.separate_file(path=input_audio_path)
    # Save the first channel of the enhanced audio to the output path
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)
    print(f'Enhanced audio saved at {output_audio_path}')

# test_function_code --------------------

def test_enhance_speech_audio():
    print("Testing enhance_speech_audio function.")
    test_input_audio_path = 'input_audio_example.wav'
    test_output_audio_path = 'output_audio_example.wav'

    # Test case: Enhancing a sample audio file
    print("Testing case [1/1] started.")
    enhance_speech_audio(test_input_audio_path, test_output_audio_path)
    assert os.path.exists(test_output_audio_path), f"Test case [1/1] failed: Enhanced audio file not found at {test_output_audio_path}"
    print("Testing case [1/1] succeeded.")

# Run the test function
test_enhance_speech_audio()