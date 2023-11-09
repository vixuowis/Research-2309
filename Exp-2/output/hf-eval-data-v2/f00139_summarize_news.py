# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news(article_text):
    """
    Summarize a given international news article text.

    Args:
        article_text (str): The text of the news article to be summarized.

    Returns:
        str: The summarized text.
    """
    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(article_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_news():
    """
    Test the summarize_news function.
    """
    article_text = "International news article text here..."
    summary = summarize_news(article_text)
    assert isinstance(summary, str), "The output must be a string."
    assert len(summary) > 0, "The output must not be an empty string."

# call_test_function_code --------------------

test_summarize_news()