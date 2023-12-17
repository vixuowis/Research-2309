# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_article(article_text, model_name):
    """
    Summarize the content of a given article text using a pre-trained mT5 model.

    Args:
        article_text (str): The text of the article to be summarized.
        model_name (str): The name of the mT5 model to use for summarization.

    Returns:
        str: A summarized version of the article.

    Raises:
        ValueError: If the article text is empty.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')
    
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )["input_ids"]
    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]
    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    
    return summary

# test_function_code --------------------



# call_test_function_line --------------------

test_summarize_article()