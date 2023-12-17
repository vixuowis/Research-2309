# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def tokenize_chinese_text(text: str) -> list:
    '''
    Tokenize a Chinese text string using a pretrained BERT model.

    Parameters:
        text (str): The Chinese text to be tokenized.

    Returns:
        list: A list of tokenized words.
    '''
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-ws')
    encoded_input = tokenizer(text, return_tensors='pt')
    return tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])

# test_function_code --------------------

def test_tokenize_chinese_text():
    print("Testing started.")
    sample_data = '我爱自然语言处理'  # An example of a Chinese sentence

    # Test case 1
    print("Testing case [1/3] started.")
    expected_tokens = ['我', '爱', '自', '然', '语', '言', '处', '理']
    actual_tokens = tokenize_chinese_text(sample_data)
    assert actual_tokens == expected_tokens, f"Test case [1/3] failed: Expected {expected_tokens}, got {actual_tokens}"

    print("Testing finished.")

# Run the test function
test_tokenize_chinese_text()