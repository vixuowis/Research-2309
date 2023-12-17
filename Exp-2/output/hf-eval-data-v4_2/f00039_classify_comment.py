# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def classify_comment(text):
    """Classify the given text as toxic or non-toxic.

    Args:
        text (str): The text or comment to classify.

    Returns:
        dict: A dictionary with the classification label and the probability.

    Raises:
        ValueError: If the text is empty or None.
    """
    if not text:
        raise ValueError('The text to classify must not be empty.')
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    result = pipeline(text)
    return result

# test_function_code --------------------

def test_classify_comment():
    print("Testing started.")

    # Test case 1: Non-toxic text
    print("Testing case [1/2] started.")
    non_toxic_result = classify_comment('Have a great day!')
    assert non_toxic_result[0]['label'] == 'LABEL_0', f"Test case [1/2] failed: Expected non-toxic, got {non_toxic_result[0]['label']}"

    # Test case 2: Toxic text
    print("Testing case [2/2] started.")
    toxic_result = classify_comment('You are so stupid!')
    assert toxic_result[0]['label'] == 'LABEL_1', f"Test case [2/2] failed: Expected toxic, got {toxic_result[0]['label']}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_comment()