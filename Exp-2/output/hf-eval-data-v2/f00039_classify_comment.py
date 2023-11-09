# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def classify_comment(text):
    """
    Classify a given text as toxic or non-toxic using a pre-trained model.

    Args:
        text (str): The text to be classified.

    Returns:
        dict: A dictionary with two keys 'LABEL_0' and 'LABEL_1'. 'LABEL_0' represents the probability of the text being non-toxic and 'LABEL_1' represents the probability of the text being toxic.
    """
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    return pipeline(text)

# test_function_code --------------------

def test_classify_comment():
    """
    Test the classify_comment function with some example texts.
    """
    example_texts = ['This is a test text.', 'You are so stupid!']
    for text in example_texts:
        result = classify_comment(text)
        assert 'LABEL_0' in result
        assert 'LABEL_1' in result
        assert 0 <= result['LABEL_0'] <= 1
        assert 0 <= result['LABEL_1'] <= 1

# call_test_function_code --------------------

test_classify_comment()