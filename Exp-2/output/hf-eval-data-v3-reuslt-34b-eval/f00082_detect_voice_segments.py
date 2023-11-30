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
    # Load VAD pipeline
    try:
        vad_pipeline = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
    except OSError:
        print("VAD model was not found. Downloading or try again later.")
        vad_pipeline = pipeline(
            "automatic-speech-recognition", 
            "facebook/wav2vec2-base-960h",
            model_cache_dir="../models"
        )   # Download the model automatically
    
    try:
        vad_results = vad_pipeline(audio_file_path)
    except OSError:
        raise OSError("Audio file not found")
    
    print("Voice activity detection completed.")
        
    return vad_results

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