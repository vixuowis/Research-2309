# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def classify_text(text):
    """
    Classify a given text as a question or a statement using a pretrained model.

    Args:
        text (str): Text to be classified.

    Returns:
        str: 'question' if the text is a question, otherwise 'statement'.
    """
    tokenizer = AutoTokenizer.from_pretrained('shahrukhx01/question-vs-statement-classifier')
    model = AutoModelForSequenceClassification.from_pretrained('shahrukhx01/question-vs-statement-classifier')

    # Tokenize the text and get predictions
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax(dim=-1).item()

    # Return the classification result
    return "question" if predicted_class == 0 else "statement"

# test_function_code --------------------

def test_classify_text():
    print("Testing classify_text function.")
    # Define test cases
    test_cases = [
        ("Is this a question?", "question"),
        ("I am making a statement.", "statement"),
        ("What time is it?", "question")
    ]

    # Test each case
    for i, (test_input, expected_output) in enumerate(test_cases, start=1):
        result = classify_text(test_input)
        assert result == expected_output, f"Test case [{i}] failed: expected '{{expected_output}}', got '{{result}}'"
        print(f"Test case [{i}] passed.")

    print("All tests passed.")