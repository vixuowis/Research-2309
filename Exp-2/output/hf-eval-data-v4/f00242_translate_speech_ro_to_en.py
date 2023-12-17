# requirements_file --------------------

!pip install -U transformers, sounddevice, soundfile 

# function_import --------------------

from transformers import pipeline
import sounddevice as sd
import soundfile as sf


# function_code --------------------

def translate_speech_ro_to_en(audio_duration: int = 5) -> bytes:
    '''
    Translates Romanian speech to English speech in real-time.

    :param audio_duration: The duration in seconds for the audio input to be recorded.
    :return: Translated English speech as audio bytes.
    '''
    # Instantiate the translation pipeline
    translator = pipeline('audio-to-audio', model='facebook/textless_sm_ro_en')

    # Record Romanian speech
    print(f'Recording for {audio_duration} seconds...')
    input_audio = sd.rec(int(audio_duration * 44100), samplerate=44100, channels=2, dtype='float32')
    sd.wait()  # Wait for the recording to finish

    # Save the recorded audio to a file
    sf.write('input_audio.wav', input_audio, 44100)

    # Translate the Romanian speech to English
    with open('input_audio.wav', 'rb') as audio_file:
        translated_audio = translator(audio_file)

    return translated_audio


# test_function_code --------------------

def test_translate_speech_ro_to_en():
    print("Testing the speech translation function.")

    # Test case: Simulate 5-second audio recording
    print("Test case: 5-second recording started.")
    output_audio = translate_speech_ro_to_en(5)
    assert isinstance(output_audio, bytes), f"Test case failed: The function did not return audio bytes."
    print("Test case: 5-second recording finished successfully.")

    print("Testing finished.")

# Run the test function
test_translate_speech_ro_to_en()
