# requirements_file --------------------

!pip install -U transformers==4.11.0.dev0

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news_article(article_text):
    """
    Summarize an international news article using a pre-trained model.

    Parameters:
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

def test_summarize_news_article():
    print("Testing summarize_news_article function.")
    sample_article = "International news article text here..."

    # Test case 1: Check if the function is providing a summary for the article.
    print("Testing case [1/1] started.")
    summary = summarize_news_article(sample_article)
    assert isinstance(summary, str) and len(summary) > 0, f"Test case [1/1] failed: the function did not return a valid summary."
    print("Testing case [1/1] completed successfully.")
    print("Testing completed.")

# Run the test function
test_summarize_news_article()