# function_import --------------------

from transformers import AutoModelForAudioToAudio
from asteroid import AudioFileProcessor

# function_code --------------------

def enhance_audio(input_audio_path: str, output_audio_path: str):
    """
    Enhance a single audio track, possibly containing dialogue, music and background noise, extracted from a video game.
    
    Args:
        input_audio_path (str): The path to the input audio file in wav format.
        output_audio_path (str): The path where the enhanced audio file will be saved in wav format.
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If the input_audio_path does not exist.
    """
    # Load the pretrained model
    audio_to_audio_model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    # Create an AudioFileProcessor object
    processor = AudioFileProcessor(audio_to_audio_model)
    # Process the input audio file and save the enhanced audio
    processor.process_file(input_audio_path, output_audio_path)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    
    Raises:
        AssertionError: If the function does not work as expected.
    """
    # Define the paths to the test input and output audio files
    test_input_audio_path = 'test_input_audio.wav'
    test_output_audio_path = 'test_output_audio.wav'
    # Call the function with the test input
    enhance_audio(test_input_audio_path, test_output_audio_path)
    # Check if the output file exists
    assert os.path.exists(test_output_audio_path), 'The output audio file does not exist.'

# call_test_function_code --------------------

test_enhance_audio()