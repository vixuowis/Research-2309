# requirements_file --------------------

!pip install -U datasets, transformers, librosa

# function_import --------------------

from transformers import pipeline
from datasets import load_dataset

# function_code --------------------

def classify_customer_voices(audio_dataset_path):
    """
    Classify and map customer voices to their identities.
    
    :param audio_dataset_path: Path to the dataset of audio files of customer voices
    :return: A dictionary mapping speaker identities to their audio files
    """
    dataset = load_dataset(audio_dataset_path)
    classifier = pipeline('audio-classification', model='superb/hubert-large-superb-sid')
    
    speaker_mapping = {}
    for audio_file in dataset['train']:
        speaker_identity = classifier(audio_file['file'], top_k=1)
        speaker_mapping[audio_file['file']] = speaker_identity
    
    return speaker_mapping

# test_function_code --------------------

def test_classify_customer_voices():
    print('Testing started.')
    dataset = load_dataset('anton-l/superb_demo', 'si', split='test')
    sample_data = dataset[0]  # Assuming the dataset is structured with 'file' field

    # Testing case: Check if the function returns a non-empty mapping
    speaker_mapping = classify_customer_voices('anton-l/superb_demo/si/test')
    assert speaker_mapping, 'Test case failed: The function did not return any speaker mapping.'

    # Testing case: Check if our sample data file is in the mapping
    assert sample_data['file'] in speaker_mapping, f"Test case failed: The sample data file is not in the mapping."

    print('Testing finished.')