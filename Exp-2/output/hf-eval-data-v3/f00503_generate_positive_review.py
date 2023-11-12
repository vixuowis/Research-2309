# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_positive_review(book_summary: str) -> str:
    """
    Generate a positive book review from a book summary using the T5-3B model.

    Args:
        book_summary (str): The summary of the book.

    Returns:
        str: A positive book review.

    Raises:
        OSError: If there is not enough disk space to download the model.
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
    """
    Test the function generate_positive_review.
    """
    book_summary = 'The book is about the adventures of a young wizard and his friends at a magic school.'
    review = generate_positive_review(book_summary)
    assert isinstance(review, str)
    assert 'positive' in review.lower()
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_positive_review()