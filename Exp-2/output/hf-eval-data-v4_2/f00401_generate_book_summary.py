# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

# function_code --------------------

def generate_book_summary(book_text):
    """
    Generates a summary of a given book text using a pre-trained T5 model.

    Args:
        book_text (str): The text of the book to summarize.

    Returns:
        dict: A dictionary containing the summary of the book.

    Raises:
        ValueError: If the provided book_text is empty.
    """
    if not book_text:
        raise ValueError('The book_text argument is empty.')

    tokenizer = T5Tokenizer.from_pretrained('pszemraj/long-t5-tglobal-base-16384-book-summary')
    summarizer = pipeline(
        'summarization',
        tokenizer=tokenizer,
        model='pszemraj/long-t5-tglobal-base-16384-book-summary',
        device=0 if torch.cuda.is_available() else -1
    )
    summary = summarizer(book_text)
    return summary[0]['summary_text']

# test_function_code --------------------

def test_generate_book_summary():
    print("Testing started.")
    
    # Test case 1: Normal book text
    print("Testing case [1/2] started.")
    book_text = "Once upon a time, there was an old library that no one visited anymore..."
    summary = generate_book_summary(book_text)
    assert summary and isinstance(summary, str), f"Test case [1/2] failed: Summary should be a string and not empty."

    # Test case 2: Empty book text
    print("Testing case [2/2] started.")
    empty_text = ""
    try:
        generate_book_summary(empty_text)
        assert False, "Test case [2/2] failed: ValueError expected for empty book text."
    except ValueError as e:
        assert str(e) == 'The book_text argument is empty.', f"Test case [2/2] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_book_summary()