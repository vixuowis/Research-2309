# function_import --------------------

from transformers import pipeline

# function_code --------------------

def detect_voice_segments(audio_file_path):
    """
    Detects voice segments in an audio file using a Voice Activity Detection (VAD) model.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        list: A list of voice segments detected in the audio file.

    Raises:
        OSError: If the specified model is not found or the audio file is not found.
    """
    
    # load a Voice Activity Detection (VAD) model
    vad_model = pipeline("voice-activity-detection")
    
    if os.path.exists(audio_file_path):
        voice_segments = [] 
        
        with open(audio_file_path, "rb") as audio_file:
            # detect the voice segments in the audio file
            results = vad_model(audio_file)
            
            for result in results:
                if result['type'] == 'v':
                    start = result['start']
                    end = result['end']
                    
                    # append a tuple of the voice segment
                    voice_segments.append((start, end))
        return voice_segments
    else:
        raise OSError(f"Audio file not found: {audio_file_path}")


# test_function_code --------------------

def test_detect_voice_segments():
    """
    Tests the detect_voice_segments function with a sample audio file.
    """
    sample_audio_file_path = 'sample_audio.wav'

    try:
        voice_segments = detect_voice_segments(sample_audio_file_path)
        assert isinstance(voice_segments, list), 'The function should return a list.'
    except OSError as e:
        print(f'Error: {e}')
    else:
        print('All Tests Passed')


# call_test_function_code --------------------

test_detect_voice_segments()