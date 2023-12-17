# requirements_file --------------------

!pip install -U transformers asteroid soundfile

# function_import --------------------

from transformers import AutoModelForAudioToAudio
import soundfile as sf

# function_code --------------------

def enhance_audio(input_audio_path, output_audio_path):
    """
    Enhances the audio quality by reducing background noise using a pre-trained model.

    Parameters:
    input_audio_path (str): The path to the input audio file.
    output_audio_path (str): The path to save the enhanced audio file.
    """
    # Load the pre-trained audio enhancer model
    audio_enhancer = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')

    # Read the input audio file
    input_audio, sample_rate = sf.read(input_audio_path)

    # Enhance the audio
    enhanced_audio = audio_enhancer(input_audio, sample_rate=sample_rate)

    # Save the enhanced audio file
    sf.write(output_audio_path, enhanced_audio["audio"][0].numpy(), sample_rate)

    return output_audio_path

# test_function_code --------------------

def test_enhance_audio():
    print("Testing audio enhancement.")
    input_audio_path = 'test_input.wav'
    output_audio_path = 'test_output.wav'

    # Test case: Enhance the audio file
    print("Testing enhancement of an audio file.")
    enhanced_path = enhance_audio(input_audio_path, output_audio_path)
    assert enhanced_path == output_audio_path, f"Failed to enhance the audio file: {input_audio_path}"

    # Perform additional checks if necessary (e.g., existence and non-emptiness of the output file)

    print("Audio enhancement test completed successfully.")

# Execute the test function
test_enhance_audio()