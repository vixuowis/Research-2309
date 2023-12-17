# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_article(article_text):
    """
    Summarize the given news article text using the mT5 multilingual XLSum model.

    Args:
        article_text (str): The text of the international news article to be summarized.
    Returns:
        str: The summary of the article.
    Raises:
        ValueError: If the article_text is empty or None.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')

    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(article_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")

    # Test case 1: Check summarization of non-empty article text
    print("Testing case [1/1] started.")
    article_text = "International news article text goes here..."
    summary = summarize_article(article_text)
    assert summary and isinstance(summary, str), f"Test case [1/1] failed: Summary is empty or not a string."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()