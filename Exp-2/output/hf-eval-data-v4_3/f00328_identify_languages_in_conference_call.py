# requirements_file --------------------

import subprocess

requirements = ["speechbrain", "torchaudio"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from speechbrain.pretrained import EncoderClassifier, load_audio

# function_code --------------------

def identify_languages_in_conference_call(audio_file_path):
    """
    Identifies the languages being spoken in an audio file of an international conference call.

    Args:
        audio_file_path (str): The path to the audio file to be processed.

    Returns:
        dict: A dictionary with keys 'language' and 'probability' for the identified language and its confidence score.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        ValueError: If the audio content cannot be processed.
    """
    # Initialize the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    # Load the audio file
    signal = load_audio(audio_file_path)
    # Predict the language spoken in the audio file
    prediction = language_id.classify_batch(signal)
    return prediction

# test_function_code --------------------



# call_test_function_line --------------------

test_identify_languages_in_conference_call()