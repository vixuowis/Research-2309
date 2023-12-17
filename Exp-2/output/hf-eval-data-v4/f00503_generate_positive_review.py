# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_positive_review(book_summary):
    """
    Generate a positive book review based on a book summary.

    Parameters:
    book_summary (str): Summary of the book.

    Returns:
    str: Generated positive book review.
    """
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    input_text = 'Write a positive review: ' + book_summary
    inputs = tokenizer(input_text, return_tensors='pt')
    outputs = model.generate(inputs)
    positive_review = tokenizer.decode(outputs[0])
    return positive_review

# test_function_code --------------------

def test_generate_positive_review():
    print("Testing generate_positive_review function.")

    # Prepare a book summary example for testing
    book_summary = "This book provides essential insights into the mechanics of business strategy."

    # Test case
    print("Testing case started.")
    positive_review = generate_positive_review(book_summary)
    assert isinstance(positive_review, str) and len(positive_review) > 0, "Test case failed: The function did not return a positive review string."
    print("Testing finished.")