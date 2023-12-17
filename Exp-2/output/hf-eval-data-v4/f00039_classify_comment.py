# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def classify_comment(text):
    """
    Classify comments as toxic or non-toxic.

    Args:
        text (str): The comment text to classify.

    Returns:
        dict: The classification results including the label and score.
    """
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    return pipeline(text)

# test_function_code --------------------

def test_classify_comment():
    print("Testing classification started.")
    # Test case 1: A non-toxic comment
    print("Testing case [1/3] started.")
    result = classify_comment('This is a great day!')
    assert result[0]['label'] == 'LABEL_0', f"Test case [1/3] failed: {result}"

    # Test case 2: A toxic comment
    print("Testing case [2/3] started.")
    result = classify_comment('You are a terrible person!')
    assert result[0]['label'] == 'LABEL_1', f"Test case [2/3] failed: {result}"

    # Test case 3: An ambiguous comment
    print("Testing case [3/3] started.")
    result = classify_comment('I'm not sure about this.')
    assert result[0]['label'] == 'LABEL_0', f"Test case [3/3] failed: {result}"
    print("Testing classification finished.")

test_classify_comment()