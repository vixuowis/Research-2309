from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model

# Function to detect voice activity in a podcast
# @param audio_file: Path to the audio file
# @return: Voice activity detection result

def detect_voice_activity(audio_file):
    # Load the pre-trained model from Hugging Face Model Hub
    model = Model.from_pretrained('pyannote/segmentation', use_auth_token='ACCESS_TOKEN_GOES_HERE')
    # Create an instance of the VoiceActivityDetection pipeline
    pipeline = VoiceActivityDetection(segmentation=model)
    # Define hyperparameters
    HYPER_PARAMETERS = {
     'onset': 0.5, 'offset': 0.5,
     'min_duration_on': 0.0,
     'min_duration_off': 0.0
    }
    # Instantiate the pipeline with the hyperparameters
    pipeline.instantiate(HYPER_PARAMETERS)
    # Process the audio file and detect voice activity
    vad = pipeline(audio_file)
    return vad