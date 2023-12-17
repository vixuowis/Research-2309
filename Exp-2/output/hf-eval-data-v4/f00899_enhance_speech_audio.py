# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

import torchaudio
from speechbrain.pretrained import WaveformEnhancement

# function_code --------------------

def enhance_speech_audio(input_audio_path, output_audio_path):
    """
    Enhance speech quality in an audio file by reducing background noise.

    Parameters:
        input_audio_path (str): The file path for the input audio to be processed.
        output_audio_path (str): The file path where the enhanced audio will be saved.

    Returns:
        None
    """
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(input_audio_path)
    torchaudio.save(output_audio_path, enhanced.unsqueeze(0).cpu(), 16000)
    print(f'Enhanced audio saved to {output_audio_path}')

# test_function_code --------------------

def test_enhance_speech_audio():
    print("Testing enhance_speech_audio function.")
    test_input = "test_input.wav"  # This should be a path to a test audio file with speech and background noise.
    test_output = "test_output.wav"  # Path where the enhanced audio will be saved.

    # Call the function with test audio file
    enhance_speech_audio(test_input, test_output)

    # Implement a test to check if the output file has been created.
    assert os.path.exists(test_output), f"Test failed: Enhanced audio file was not saved to {test_output}."

    print("Test passed: Enhanced audio file saved.")

# Run the test function
test_enhance_speech_audio()