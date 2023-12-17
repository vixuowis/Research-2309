# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import AutoModelForAudioToAudio
from asteroid import AudioFileProcessor

# function_code --------------------

def enhance_audio_track(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhances an audio track by reducing noise and improving quality.

    Args:
        input_audio_path (str): The file path for the input audio in wav format.
        output_audio_path (str): The file path where the enhanced audio will be saved in wav format.

    Returns:
        None: The function saves the enhanced audio to the specified path.

    Raises:
        FileNotFoundError: If the input_audio_path does not exist.
        Exception: If there is an error in processing the audio file.
    """
    try:
        # Load pre-trained model for enhancing audio
        model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
        # Initialize audio file processor with the loaded model
        processor = AudioFileProcessor(model)

        # Check if the input audio file exists
        if not os.path.exists(input_audio_path):
            raise FileNotFoundError(f"Input audio file not found: {input_audio_path}")

        # Process the audio file and save the enhanced version
        processor.process_file(input_audio_path, output_audio_path)
    except Exception as e:
        raise Exception(f"Error in processing the audio file: {e}")


# test_function_code --------------------

def test_enhance_audio_track():
    print("Testing started.")
    input_audio_path = 'test_input_audio.wav'
    output_audio_path = 'test_enhanced_audio.wav'

    # Test case 1: File exists and enhancement is successful
    print("Testing case [1/3] started.")
    assert os.path.isfile(input_audio_path), f"Test case [1/3] failed: Input audio file does not exist."
    enhance_audio_track(input_audio_path, output_audio_path)
    assert os.path.isfile(output_audio_path), f"Test case [1/3] failed: Output audio file was not created."

    # Test case 2: File does not exist
    print("Testing case [2/3] started.")
    non_existent_path = 'non_existent_audio.wav'
    try:
        enhance_audio_track(non_existent_path, output_audio_path)
    except FileNotFoundError:
        assert True
    else:
        assert False, f"Test case [2/3] failed: No exception raised for nonexistent input file."

    # Test case 3: Process fails
    print("Testing case [3/3] started.")
    # Simulate an error in processing
    with unittest.mock.patch('asteroid.AudioFileProcessor.process_file', side_effect=Exception('Processing error')):
        try:
            enhance_audio_track(input_audio_path, output_audio_path)
        except Exception as e:
            assert 'Processing error' in str(e), f"Test case [3/3] failed: Incorrect exception message."

    print("Testing finished.")


# call_test_function_line --------------------

test_enhance_audio_track()