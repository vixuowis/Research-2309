# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_article(sequence, candidate_labels):
    """
    Classifies a given text sequence into a category using a zero-shot classification model.

    Args:
        sequence (str): The text article to be classified.
        candidate_labels (list(str)): A list of potential categories to classify the article into.

    Returns:
        dict: A dictionary containing labels and their corresponding scores.

    Raises:
        ValueError: If sequence or candidate_labels are not provided.
    """
    if not sequence or not candidate_labels:
        raise ValueError('Input sequence or candidate_labels are missing.')
    zero_shot_classifier = pipeline('zero-shot-classification', model='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')
    classification_output = zero_shot_classifier(sequence, candidate_labels)
    return classification_output

# test_function_code --------------------

def test_classify_article():
    print('Testing started.')
    # Define the test cases
    test_cases = [
        {
            'sequence': 'Angela Merkel ist eine Politikerin in Deutschland und Vorsitzende der CDU.',
            'candidate_labels': ['politics', 'economy', 'entertainment', 'environment'],
            'expected_label': 'politics'
        },
        # Add more test cases here
    ]
    
    # Iterate over test cases
    for i, test_case in enumerate(test_cases):
        print(f'Testing case [{i+1}/{len(test_cases)}] started.')
        output = classify_article(test_case['sequence'], test_case['candidate_labels'])
        assert output['labels'][0] == test_case['expected_label'], f'Test case [{i+1}/{len(test_cases)}] failed: Expected label and output label do not match.'
        print(f'Test case [{i+1}/{len(test_cases)}] passed.')
    print('Testing finished.')

# call_test_function_line --------------------

test_classify_article()