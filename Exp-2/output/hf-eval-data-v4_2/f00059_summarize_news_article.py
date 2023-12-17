# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_news_article(input_text):
    """
    Summarizes a news article using the Pegasus model.

    Args:
        input_text (str): The text of the news article to summarize.

    Returns:
        str: The summarized version of the news article.

    Raises:
        ValueError: If the input_text is empty.
    """
    if not input_text:
        raise ValueError('The input_text is empty')
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing started.")
    sample_data = "This is a sample news article text to test the summarization function. The content of the article is not relevant, only the fact that the function is capable of summarizing the text matters in this testing scenario."

    # Testing case 1: Summarize a non-empty news article
    print("Testing case [1/1] started.")
    summary = summarize_news_article(sample_data)
    assert isinstance(summary, str) and len(summary) > 0, f"Test case [1/1] failed: The summary is not a non-empty string."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_news_article()