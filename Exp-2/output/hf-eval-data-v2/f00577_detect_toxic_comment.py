# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def detect_toxic_comment(message):
    """
    Detect if there are any harmful messages in a chat room.

    Args:
        message (str): The message to be classified.

    Returns:
        dict: The classification results for the message as toxic or non-toxic.
    """
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    toxicity_result = pipeline(message)
    return toxicity_result

# test_function_code --------------------

def test_detect_toxic_comment():
    """
    Test the function detect_toxic_comment.
    """
    message = 'This is a test text.'
    result = detect_toxic_comment(message)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'label' in result, 'The result should have a label.'
    assert 'score' in result, 'The result should have a score.'

# call_test_function_code --------------------

test_detect_toxic_comment()