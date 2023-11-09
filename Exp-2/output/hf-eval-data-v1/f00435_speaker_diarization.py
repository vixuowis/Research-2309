from pyannote.audio import Pipeline
import os

# Function to perform speaker diarization
# Speaker diarization is the task of determining 'who spoke when' in an audio or video recording.
# This function uses the 'philschmid/pyannote-speaker-diarization-endpoint' pre-trained model from pyannote.audio
# to perform speaker diarization.
def speaker_diarization(audio_file):
    # Check if the audio file exists
    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f'{audio_file} not found')
    
    # Load the pre-trained model
    diarization_pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')
    
    # Perform speaker diarization
    diarization = diarization_pipeline(audio_file)
    
    # Write the result to an RTTM file
    with open('output_audio.rttm', 'w') as rttm:
        diarization.write_rttm(rttm)
    
    return 'output_audio.rttm'