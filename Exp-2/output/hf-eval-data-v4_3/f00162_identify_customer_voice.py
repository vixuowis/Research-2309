# requirements_file --------------------

import subprocess

requirements = ["datasets", "transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from datasets import load_dataset
from transformers import pipeline

# function_code --------------------

def identify_customer_voice(dataset_name, model_name='superb/hubert-large-superb-sid', top_k=5):
    """
    Identifies customer voices from an audio dataset using a pretrained model.

    Args:
        dataset_name: The name of the dataset containing customer voices.
        model_name: The Hugging Face model identifier to be used for audio classification.
        top_k: The number of top predictions to consider for speaker identification.

    Returns:
        A dictionary mapping speaker identities to their corresponding audio files.

    Raises:
        ValueError: If the dataset is empty or not found.

    """
    # Load dataset
    dataset = load_dataset(dataset_name)
    if not dataset:
        raise ValueError('The dataset is empty or not found.')

    # Setup classifier
    classifier = pipeline('audio-classification', model=model_name)

    # Create speaker-to-voice mapping
    speaker_id_map = {}
    for audio_file in dataset:
        speaker_identity = classifier(audio_file['file'], top_k=top_k)
        # Store the identity mapping
        speaker_id_map[audio_file['file']] = speaker_identity

    return speaker_id_map

# test_function_code --------------------

def test_identify_customer_voice():
    print("Testing started.")
    dataset_name = 'anton-l/superb_demo'
    sample_data = load_dataset(dataset_name, split='test')
    assert sample_data is not None, "Test case failed: Sample data not found."

    print("Testing case [1/1] started.")
    speaker_id_map = identify_customer_voice(dataset_name)
    assert speaker_id_map, "Test case failed: No speaker identification mapping found."
    print("Testing finished.")

# call_test_function_line --------------------

test_identify_customer_voice()