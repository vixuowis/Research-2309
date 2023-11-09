# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_positive_review(book_summary):
    """
    This function converts a book summary into a positive book review using the T5-3B model from Hugging Face Transformers.

    Args:
        book_summary (str): The summary of the book.

    Returns:
        str: A positive review of the book.
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
    This function tests the generate_positive_review function.
    It uses a sample book summary and checks if the output is a string.
    """
    book_summary = 'This is a book about the adventures of a young wizard.'
    review = generate_positive_review(book_summary)
    assert isinstance(review, str), 'The review should be a string.'

# call_test_function_code --------------------

test_generate_positive_review()