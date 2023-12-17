# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# function_code --------------------

def classify_text(sequence: str, candidate_labels: list) -> str:
    """Classify a text message into one of the provided categories using zero-shot classification.

    Args:
        sequence (str): The text message to classify.
        candidate_labels (list): The list of categories to classify the sequence into.

    Returns:
        str: The category that the text message is most likely to belong to.

    Raises:
        ValueError: If candidate_labels is empty.

    """
    if not candidate_labels:
        raise ValueError("candidate_labels should not be empty.")

    # Load the pre-trained NLI model
    nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli')

    # Compute probabilities for each label
    probs_list = []
    for label in candidate_labels:
        hypothesis = f'This example is {label}.'
        inputs = tokenizer(sequence, hypothesis, return_tensors='pt', truncation=True)
        logits = nli_model(**inputs)[0]
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1].item()
        probs_list.append(prob_label_is_true)

    # Determine the category with the highest probability
    category_index = probs_list.index(max(probs_list))
    return candidate_labels[category_index]

# test_function_code --------------------

def test_classify_text():
    print("Testing started.")

    # Test case 1: Typical text message
    print("Testing case [1/3] started.")
    assert classify_text("I love exploring new cities and cultures.", ['travel', 'cooking', 'sports']) == 'travel', "Test case [1/3] failed: Expected 'travel'."

    # Test case 2: Empty candidate labels
    print("Testing case [2/3] started.")
    try:
        classify_text("Cooking is my way to unwind and be creative.", [])
        assert False, "Test case [2/3] failed: ValueError not raised for empty candidate_labels."
    except ValueError as e:
        assert str(e) == 'candidate_labels should not be empty.', "Test case [2/3] failed: Incorrect ValueError message."

    # Test case 3: Text message with multiple possible categories
    print("Testing case [3/3] started.")
    assert classify_text("The rhythm and flow of the music make me want to dance.", ['music', 'dancing', 'reading']) == 'dancing', "Test case [3/3] failed: Expected 'dancing'."

    print("Testing finished.")

# call_test_function_line --------------------

test_classify_text()