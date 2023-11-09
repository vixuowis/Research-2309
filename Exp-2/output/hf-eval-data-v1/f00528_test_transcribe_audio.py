import os
import urllib.request

# Test function for transcribe_audio
# This function downloads a sample audio file, transcribes it using the transcribe_audio function, and prints the transcription.
def test_transcribe_audio():
    # Download a sample audio file
    url = 'https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav'
    filename = 'sample.wav'
    urllib.request.urlretrieve(url, filename)
    # Transcribe the audio file
    transcription = transcribe_audio(filename)
    # Print the transcription
    print(transcription)
    # Remove the sample audio file
    os.remove(filename)

# Call the test function
test_transcribe_audio()