# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tokenize_chinese_text(text):
    """
    Tokenizes a text using a pretrained Chinese BERT model.

    Args:
        text (str): The text to be tokenized.

    Returns:
        list: A list of tokenized words.

    Raises:
        ValueError: If the text input is not a string.
    """
    if not isinstance(text, str):
        raise ValueError('Input must be a string.')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws')
    tokens = tokenizer.tokenize(text)
    return tokens

# test_function_code --------------------

def test_tokenize_chinese_text():
    print("Testing started.")

    # Test case 1: Normal string
    print("Testing case [1/3] started.")
    sample_text = '今天天气不错'
    expected_tokens = ['今天', '天气', '不', '错']
    assert tokenize_chinese_text(sample_text) == expected_tokens, f"Test case [1/3] failed: Expected {expected_tokens}, got {tokenize_chinese_text(sample_text)}"

    # Test case 2: Empty string
    print("Testing case [2/3] started.")
    sample_text = ''
    assert tokenize_chinese_text(sample_text) == [], f"Test case [2/3] failed: Expected [], got {tokenize_chinese_text(sample_text)}"

    # Test case 3: Non-string input
    print("Testing case [3/3] started.")
    sample_text = None
    try:
        tokenize_chinese_text(sample_text)
        assert False, "Test case [3/3] failed: ValueError expected."
    except ValueError as e:
        assert str(e) == 'Input must be a string.', f"Test case [3/3] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_tokenize_chinese_text()