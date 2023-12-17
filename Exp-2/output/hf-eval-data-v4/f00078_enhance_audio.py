# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_audio_path, output_audio_path):
    """
    Enhances an audio file by reducing noise and reverberation using the Sepformer model.

    Args:
    input_audio_path (str): The path to the input noisy audio file.
    output_audio_path (str): The path where the enhanced audio file will be saved.

    Returns:
    bool: True if the audio was enhanced and saved successfully, False otherwise.
    """
    try:
        # Load the pretrained Sepformer model
        model = separator.from_hparams(source='speechbrain/sepformer-whamr-enhancement', savedir='pretrained_models/sepformer-whamr-enhancement')

        # Enhance the audio
        est_sources = model.separate_file(path=input_audio_path)

        # Save the enhanced audio to the output path
        torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)

        return True
    except Exception as e:
        print(f'Error occurred: {e}')
        return False

# test_function_code --------------------

def test_enhance_audio():
    print("Testing enhance_audio function.")

    # Assumed existing audio path for testing
    test_input_audio_path = 'test_input_audio.wav'

    # Assumed path for saving the output of the enhanced audio
    test_output_audio_path = 'test_output_audio.wav'

    # Test case 1: Check if the function returns True on successful enhancement
    print("Testing case [1/1] started.")
    result = enhance_audio(test_input_audio_path, test_output_audio_path)
    assert result, "Test case [1/1] failed: The function did not return True."

    print("Testing finished successfully.")

# Run the test function
test_enhance_audio()