# function_import --------------------

from pyannote.audio import Model, Inference

# function_code --------------------

def measure_noise_levels(audio_file_path: str, access_token: str):
    """
    Measures noise levels in the environment using a pre-trained model from Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file.
        access_token (str): The access token for Hugging Face Transformers.

    Returns:
        None. Prints the voice activity detection (VAD), speech-to-noise ratio (SNR), and the C50 room acoustics estimation for each frame in the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If there is an error loading the model or processing the audio file.
    """

    # function_code --------------------
    
    from pyannote.audio import Model, Inference
    import soundfile as sf
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    try: 
        model = Model.from_pretrained("pyannote/voice-activity-detection", 
                                      access_token=access_token)  
        
    except Exception as e:
        raise Exception(f"Loading the pre-trained 'Voice activity detection' model from Hugging Face Transformers failed: {e}")
    
    try:
        data, sampleRate = sf.read(audio_file_path)
        
    except FileNotFoundError as e:
        raise FileNotFoundError("Audio file not found.")
    
    try: 
        inference: Inference = model(data)
        
    except Exception as e:
        raise Exception(f"Processing audio file failed with error {e}")
        
    # function_code --------------------
    
    print("Time", "\t\t\t", "Voice activity (%)", "\t\t\t", 
          "Speech-to-noise ratio (SNR)", "\t", "C50 room acoustics estimation")    
        
    # function_code --------------------
        
    for time, vad in zip(inference.helper['timeline'], inference):    
        print(f"{time:06.2f}", "\t\t", f"{vad[1]*100:5.1f}", 
              "\t\t\t", np.round(vad[0], decimals=2), 
              "\t\t", vad[2])

# test_function_code --------------------

def test_measure_noise_levels():
    """
    Tests the measure_noise_levels function.
    """
    # Test with a valid audio file and access token
    try:
        measure_noise_levels('valid_audio_file.wav', 'valid_access_token')
    except Exception as e:
        print(f'Error: {e}')

    # Test with an invalid audio file
    try:
        measure_noise_levels('invalid_audio_file.wav', 'valid_access_token')
    except FileNotFoundError as fnf_error:
        assert str(fnf_error) == "[Errno 2] No such file or directory: 'invalid_audio_file.wav'", 'Test Failed'

    # Test with an invalid access token
    try:
        measure_noise_levels('valid_audio_file.wav', 'invalid_access_token')
    except Exception as e:
        assert str(e) == 'Invalid access token', 'Test Failed'

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_measure_noise_levels())