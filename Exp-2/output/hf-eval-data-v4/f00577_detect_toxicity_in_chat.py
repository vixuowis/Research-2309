# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline

# function_code --------------------

def detect_toxicity_in_chat(message: str) -> dict:
    """
    This function uses a pre-trained model from Hugging Face to detect harmful messages.
    
    Args:
        message (str): The message text to analyze for toxicity.
    
    Returns:
        dict: The classification result indicating if the message is toxic or not.
    """
    model_path = 'martin-ha/toxic-comment-model'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer)
    classification_result = pipeline(message)
    return classification_result

# test_function_code --------------------

def test_detect_toxicity_in_chat():
    print("Testing detect_toxicity_in_chat function.")
    test_messages = [
        ("I love this.", False),
        ("You are an idiot!", True),
        ("Have a great day!", False)
    ]

    for i, (message, expected_toxic) in enumerate(test_messages, start=1):
        result = detect_toxicity_in_chat(message)
        assert ('label' in result and 'score' in result), f"Test case [{i}/3] failed: Function should return a dictionary with 'label' and 'score'."
        is_toxic = result['label'] == 'LABEL_1'
        assert (is_toxic == expected_toxic), f"Test case [{i}/3] failed: Expected toxic='{{expected_toxic}}', got '{{is_toxic}}'."
        print(f"Test case [{i}/3] passed.")
    print("Testing completed.")