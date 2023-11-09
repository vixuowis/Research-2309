from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection

# Function to detect overlapped speech in a conference call
# @param conference_call_audio_file: The audio file of the conference call
# @return overlap_results: The detected overlapped speech data

def detect_overlapped_speech(conference_call_audio_file):
    # Load the pre-trained 'pyannote/segmentation' model from the Hugging Face Model Hub
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token='ACCESS_TOKEN_GOES_HERE')
    # Instantiate a OverlappedSpeechDetection pipeline using the loaded model
    pipeline = OverlappedSpeechDetection(segmentation=model)
    # Set hyperparameters for the OverlappedSpeechDetection pipeline
    HYPER_PARAMETERS = {
        'onset': 0.5,
        'offset': 0.5,
        'min_duration_on': 0.0,
        'min_duration_off': 0.0
    }
    # Instantiate the pipeline with the set hyperparameters
    pipeline.instantiate(HYPER_PARAMETERS)
    # Process the conference call audio file using the instantiated pipeline
    overlap_results = pipeline(conference_call_audio_file)
    # Return the detected overlapped speech data
    return overlap_results