# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def detect_harmful_messages(text):
    """
    Detects if there are any harmful messages in a chat room.

    Args:
        text (str): The text message to be classified.

    Returns:
        dict: Containing the classification results as 'toxic' or 'non-toxic'.

    Raises:
        ValueError: If the text is empty.
    """
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    if not text:
        raise ValueError("The text message is empty.")

    results = pipeline(text)
    return results

# test_function_code --------------------

def test_detect_harmful_messages():
    print("Testing started.")

    # Test case 1: Check non-toxic message
    print("Testing case [1/3] started.")
    assert detect_harmful_messages('Have a great day!')[0]['label'] == 'LABEL_0', "Test case [1/3] failed: Non-toxic message classified as toxic."

    # Test case 2: Check toxic message
    print("Testing case [2/3] started.")
    assert detect_harmful_messages('You are so stupid!')[0]['label'] == 'LABEL_1', "Test case [2/3] failed: Toxic message not identified."

    # Test case 3: Check empty message
    print("Testing case [3/3] started.")
    try:
        detect_harmful_messages('')
        assert False, "Test case [3/3] failed: Empty message did not raise ValueError."
    except ValueError as e:
        assert str(e) == "The text message is empty.", "Test case [3/3] failed: Incorrect ValueError message."

    print("Testing finished.")

# call_test_function_line --------------------

test_detect_harmful_messages()