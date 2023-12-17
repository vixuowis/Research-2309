# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def summarize_article_in_french(article_text: str) -> str:
    """
    Summarize the content of an article in French.

    Args:
        article_text (str): The article text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the article_text is empty.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')

    model_name = 'plguillou/t5-base-fr-sum-cnndm'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = f'summarize: {article_text}'
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    summary_ids = model.generate(input_tokens)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

# test_function_code --------------------

def test_summarize_article_in_french():
    print('Testing started.')
    article_text = 'Selon un rapport récent, les constructeurs automobiles prévoient...'
    expected_summary = 'Les constructeurs automobiles accélèrent la production...'

    print('Testing case [1/1] started.')
    summary = summarize_article_in_french(article_text)
    assert summary == expected_summary, f'Test case [1/1] failed: Expected {expected_summary}, got {summary}'
    print('Testing finished.')

# call_test_function_line --------------------

test_summarize_article_in_french()