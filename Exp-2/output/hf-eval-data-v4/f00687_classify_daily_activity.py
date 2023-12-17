# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# function_code --------------------

def classify_daily_activity(text: str, categories: list) -> str:
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    result = classifier(text, categories)
    return result['labels'][0]

# test_function_code --------------------

def test_classify_daily_activity():
    print("Testing classify_daily_activity function.")
    sample_text = "I spent hours in the kitchen trying a new recipe."
    categories = ['travel', 'cooking', 'dancing']
    expected_output = 'cooking'
    output = classify_daily_activity(sample_text, categories)
    assert output == expected_output, f"Expected category '{{expected_output}}' but got '{{output}}'"
    print("classify_daily_activity function test passed")

test_classify_daily_activity()