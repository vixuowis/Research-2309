# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

# function_code --------------------

def summarize_book(book_text):
    """
    Summarize a book using a pre-trained model from Hugging Face.

    Args:
        book_text (str): The content of the book to be summarized.

    Returns:
        str: The summary of the book.
    """
    tokenizer = T5Tokenizer.from_pretrained('pszemraj/long-t5-tglobal-base-16384-book-summary')
    summarizer = pipeline(
        'summarization',
        tokenizer=tokenizer,
        model='pszemraj/long-t5-tglobal-base-16384-book-summary',
        device=0 if torch.cuda.is_available() else -1)
    summary = summarizer(book_text)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_book():
    """
    Test the function summarize_book.
    """
    book_text = 'This is a test book. It has a beginning, a middle, and an end.'
    summary = summarize_book(book_text)
    assert isinstance(summary, str), 'The result should be a string.'
    assert len(summary) > 0, 'The summary should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_book()