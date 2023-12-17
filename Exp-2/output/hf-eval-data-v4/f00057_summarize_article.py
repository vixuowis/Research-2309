# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(article_text):
    # Instantiate the summarization pipeline from Hugging Face Transformers
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

    # Summarize the provided article text
    summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)

    # Return the summary text
    return summary[0]['summary_text']

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")

    # Test article
    article = "Apple Inc. reported its quarterly earnings results yesterday. The company posted a record-breaking revenue..."

    # Expected summary
    expected_summary = "Apple posted record-breaking Q1 2022 revenue of $123.9 billion..."

    # Test case
    print("Testing summarize_article function.")
    summary = summarize_article(article)
    assert summary == expected_summary, f"Test failed: expected {{expected_summary}}, got {{summary}}"
    print("Test passed.")

    print("Testing finished.")