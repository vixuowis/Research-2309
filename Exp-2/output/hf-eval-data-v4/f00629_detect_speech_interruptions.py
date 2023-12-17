# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection

# function_code --------------------

def detect_speech_interruptions(audio_file_path, access_token):
    """
    Detects speech interruptions in a conference call audio file.

    Parameters:
    - audio_file_path (str): The path to the conference call audio file.
    - access_token (str): The access token for using the pre-trained model from Hugging Face.

    Returns:
    - List[Tuple[float, float]]: A list of tuples where each tuple represents the start and end time of an overlapped speech segment indicating an interruption.
    """
    # Load the pre-trained 'pyannote/segmentation' model from Hugging Face Model Hub
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token=access_token)

    # Instantiate the OverlappedSpeechDetection pipeline with the loaded model
    pipeline = OverlappedSpeechDetection(segmentation=model)

    # Set hyperparameters for the OverlappedSpeechDetection pipeline
    HYPER_PARAMETERS = {
        'onset': 0.5,
        'offset': 0.5,
        'min_duration_on': 0.0,
        'min_duration_off': 0.0
    }

    # Instantiate the pipeline with these parameters
    pipeline.instantiate(HYPER_PARAMETERS)

    # Process the audio file to detect overlapped speech segments
    overlap_results = pipeline(audio_file_path)

    # Convert segment predictions to a list of start-end time tuples
    interruptions = [(segment.start, segment.end) for segment in overlap_results.get_timeline()]

    return interruptions

# test_function_code --------------------

def test_detect_speech_interruptions():
    print("Testing started.")
    # Assuming 'sample_conference_call.wav' is a sample audio file in the current directory for testing
    sample_audio_file = "sample_conference_call.wav"
    
    # Replace 'YOUR_ACCESS_TOKEN' with the actual access token for Hugging Face
    access_token = "YOUR_ACCESS_TOKEN"
    
    # Testing case [1/1]: Detecting speech interruptions in a sample conference call audio.
    print("Testing case [1/1] started.")
    interruptions = detect_speech_interruptions(sample_audio_file, access_token)
    assert interruptions, f"Test case [1/1] failed: No interruptions detected."

    print("Testing finished.")

# Run the test function
test_detect_speech_interruptions()