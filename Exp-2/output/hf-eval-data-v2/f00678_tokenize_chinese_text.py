# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tokenize_chinese_text(text):
    """
    Tokenizes a given Chinese text using the 'ckiplab/bert-base-chinese-ws' pretrained model.

    Args:
        text (str): The Chinese text to be tokenized.

    Returns:
        A list of tokens.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws')
    tokens = tokenizer.tokenize(text)
    return tokens

# test_function_code --------------------

def test_tokenize_chinese_text():
    """
    Tests the 'tokenize_chinese_text' function by tokenizing a sample Chinese text.
    """
    sample_text = '我爱自然语言处理'
    tokens = tokenize_chinese_text(sample_text)
    assert isinstance(tokens, list), 'The output should be a list of tokens.'
    assert len(tokens) > 0, 'The list of tokens should not be empty.'

# call_test_function_code --------------------

test_tokenize_chinese_text()