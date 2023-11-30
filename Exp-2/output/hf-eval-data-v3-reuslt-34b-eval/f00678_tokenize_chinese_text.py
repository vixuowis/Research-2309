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

    # Load Bert tokenizer for Chinese text
    bert_tokenizer = BertTokenizerFast.from_pretrained('ckiplab/bert-base-chinese-ws')
    
    # Tokenize the input text and add '[CLS]' at the beginning of the sequence, as well as adding a '[SEP]' token at the end
    tokenized_text = bert_tokenizer(text,
                                    return_tensors='pt',
                                    truncation=True, 
                                    max_length=512, 
                                    padding="max_length", 
                                    add_special_tokens=True)["input_ids"][0] # Extracts the PyTorch tensor returned by BertTokenizerFast.from_pretrained().
    
    # Convert tokenized text from a PyTorch tensor to a list of strings
    tokenized_text = [bert_tokenizer.decode([token]) for token in tokenized_text]
    
    return tokenized_text


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