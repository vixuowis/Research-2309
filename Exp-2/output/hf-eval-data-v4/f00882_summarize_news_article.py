# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_news_article(news_article):
    """
    Summarize a news article using the Pegasus model from Hugging Face Transformers.

    Parameters:
    news_article (str): The news article to summarize.

    Returns:
    str: The summarized version of the news article.
    """
    # Initialize the model and tokenizer
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    # Encode the article using the tokenizer
    inputs = tokenizer.encode(news_article, truncation=True, return_tensors='pt')

    # Generate summary ids
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

    # Decode and return the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing summarize_news_article function.")
    example_article = "The quick brown fox jumps over the lazy dog. This is used as a test phrase for typing programs.
"
    expected_summary = "The quick brown fox jumps over the lazy dog."

    # Test if the summarized text is a substring of the example article
    print("Testing case [1/1] started.")
    summary = summarize_news_article(example_article)
    assert expected_summary in summary, f"Test case [1/1] failed: summarized text is not a correct summary"
    print("Testing case [1/1] passed.")

    print("Testing finished.")

# Running the test function
test_summarize_news_article()