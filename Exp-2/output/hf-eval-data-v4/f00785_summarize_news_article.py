# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_news_article(article_text):
    # Initialize the summarizer model
    summarizer = pipeline('summarization', model='facebook/bart-large-cnn')

    # Generate summary for the provided article text
    summary = summarizer(article_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    return summary


# test_function_code --------------------

def test_summarize_news_article():
    print("Testing summarize_news_article function.")

    # Example news article
    article_text = 'Breaking news: multiple sources report significant advances in machine learning ...'
    expected_summary = 'Machine learning '  # Simplified expected summary

    # Test the summarization function
    summary = summarize_news_article(article_text)

    # Check if the summary is as expected
    assert summary.startswith(expected_summary), f"Test failed: Summary '{summary}' does not match expected start '{expected_summary}'"

    print("Testing completed successfully.")

# Run the test function
test_summarize_news_article()
