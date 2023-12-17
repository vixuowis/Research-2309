# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_news_article(news_article):
    """Summarizes a news article using the Pegasus Transformer model.

    Args:
        news_article (str): The news article to summarize.

    Returns:
        str: A summary of the news article.

    Raises:
        ValueError: If the news article is empty or None.
    """
    if not news_article:
        raise ValueError('The news article is empty or None.')
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(news_article, return_tensors='pt', truncation=True, max_length=1024)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing started.")
    sample_news_article = "..."  # Replace with sample news article text

    # Test case 1: Valid news article
    print("Testing case [1/1] started.")
    summary = summarize_news_article(sample_news_article)
    assert isinstance(summary, str) and summary, "Test case [1/1] failed: The summary is not a non-empty string."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_news_article()