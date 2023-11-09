from pyannote.audio import Pipeline
import os

# Function to perform speaker diarization
# @param audio_file: Path to the audio file
# @return: Diarization object

def speaker_diarization(audio_file):
    # Ensure the audio file exists
    if not os.path.exists(audio_file):
        raise ValueError(f'Audio file does not exist: {audio_file}')
    
    # Load the pretrained pipeline
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')
    
    # Perform diarization
    diarization = pipeline(audio_file)
    
    return diarization