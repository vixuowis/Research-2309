# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news_article(article_text):
    """
    Summarizes a news article text using a pre-trained multilingual transformer model.

    Args:
        article_text (str): The article text to summarize.

    Returns:
        str: The summary of the news article.

    Raises:
        ValueError: If the provided article_text is not a string or empty.

    """
    if not isinstance(article_text, str) or not article_text:
        raise ValueError('The article_text must be a non-empty string.')

    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing started.")
    article = "This is a test article to demonstrate the functionality of the summarize_news_article function."

    # Testing case 1
    print("Testing case [1/1] started.")
    summary = summarize_news_article(article)
    assert summary, "Test case [1/1] failed: The function did not return a summary."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_news_article()