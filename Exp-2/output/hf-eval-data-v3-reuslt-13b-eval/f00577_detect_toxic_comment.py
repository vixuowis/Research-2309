# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def detect_toxic_comment(message):
    """
    Detect if a message is toxic or not using a pre-trained model from Hugging Face Transformers.

    Args:
        message (str): The message to be classified.

    Returns:
        dict: A dictionary containing the classification results.
    """
    
    # Load the model and tokenizer from local disk
    # You can download them here: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads
    model = AutoModelForSequenceClassification.from_pretrained("../saved_files")
    tokenizer = AutoTokenizer.from_pretrained("../saved_files")

    # Instantiate the pipeline
    toxicity_classifier = TextClassificationPipeline(model=model,
                                                      tokenizer=tokenizer)

    # Classify the message
    results = toxicity_classifier(message)[0]

    # Return a dictionary containing the classification results
    return {"toxic": bool(results["label"])}

# test_function_code --------------------

def test_detect_toxic_comment():
    """
    Test the function detect_toxic_comment.
    """
    message1 = 'This is a test text.'
    message2 = 'You are so stupid!'
    message3 = 'Have a nice day!'
    result1 = detect_toxic_comment(message1)
    result2 = detect_toxic_comment(message2)
    result3 = detect_toxic_comment(message3)
    assert isinstance(result1, list), 'The result should be a list.'
    assert isinstance(result2, list), 'The result should be a list.'
    assert isinstance(result3, list), 'The result should be a list.'
    print('All Tests Passed')


# call_test_function_code --------------------

test_detect_toxic_comment()