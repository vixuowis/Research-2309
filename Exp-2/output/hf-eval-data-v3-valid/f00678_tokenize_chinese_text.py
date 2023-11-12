# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tokenize_chinese_text(text):
    """
    Tokenizes Chinese text using the 'ckiplab/bert-base-chinese-ws' pretrained model.

    Args:
        text (str): The Chinese text to be tokenized.

    Returns:
        List[str]: The tokenized text.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws')
    tokens = tokenizer.tokenize(text)
    return tokens

# test_function_code --------------------

def test_tokenize_chinese_text():
    """
    Tests the tokenize_chinese_text function with some sample Chinese text.
    """
    sample_text = '我爱自然语言处理'
    tokens = tokenize_chinese_text(sample_text)
    assert isinstance(tokens, list)
    assert all(isinstance(token, str) for token in tokens)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_tokenize_chinese_text()