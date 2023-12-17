# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import torch

# function_code --------------------

def generate_summary(book_text):
    """
    Generate a SparkNotes-style summary for a given book text.

    Parameters:
        book_text: str
            The text of the book to summarize.

    Returns:
        str
            The generated summary of the book.
    """
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

def test_generate_summary():
    print("Testing generate_summary function.")
    sample_text = "Once upon a time, a long text that needs to be summarized was given."
    expected_output_start = 'Once upon a time,'

    # Testing the function
    print("Testing with sample text.")
    summary = generate_summary(sample_text)
    assert summary.startswith(expected_output_start), f"Test failed: Summary did not start with expected text."
    print("Test passed: Summary starts with expected text.")

# Run the test
print("Running tests for generate_summary function.")
try:
    test_generate_summary()
    print("All tests passed successfully.")
except AssertionError as e:
    print(str(e))