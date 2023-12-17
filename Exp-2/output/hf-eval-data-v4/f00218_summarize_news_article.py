# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news_article(article_text):
    """
    Summarize a news article using a pre-trained multilingual model.

    Parameters:
        article_text (str): The text of the news article to be summarized.

    Returns:
        str: The summarized text.
    """
    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(article_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing summarize_news_article function.")

    # Test case 1: Short news article
    article_text = "Researchers have found a new species of octopus in the Pacific Ocean."
    expected_summary = "New species of octopus found."
    summary = summarize_news_article(article_text)
    assert summary == expected_summary, f"Test case failed: expected {expected_summary}, got {summary}"

    # Test case 2: Long news article
    article_text = "The tech industry has seen a significant increase in demand for cybersecurity experts due to the rise in cyber attacks. Companies are investing more in securing their digital assets."
    expected_part = "tech industry ... cybersecurity experts"
    summary = summarize_news_article(article_text)
    assert expected_part in summary, f"Test case failed: expected summary to contain {expected_part}"

    print("All test cases passed.")