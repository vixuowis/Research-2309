# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def predict_masked_token(text):
    """
    Predicts the masked token in a Chinese sentence using the 'bert-base-chinese' model.

    Args:
        text (str): A Chinese sentence with one token masked (indicated by <mask>).

    Returns:
        str: A Chinese sentence with the masked token replaced by the model's prediction.

    Raises:
        ValueError: If the input text does not contain exactly one masked token.
    """
    if text.count('<mask>') != 1:
        raise ValueError("The input text must contain exactly one masked token.")

    fill_mask = pipeline('fill-mask', model='bert-base-chinese')
    result = fill_mask(text)

    # Assuming the top prediction is the most appropriate one.
    filled_text = result[0]['sequence']
    return filled_text

# test_function_code --------------------

def test_predict_masked_token():
    print("Testing started.")
    # 定义测试用例
    test_cases = [
        {'input': '我们很高兴与您合作，希望我们的<mask>能为您带来便利。', 'expected_mask': '<mask>', 'expected_output_contains': '服务'},
        {'input': '我喜欢在<mask>阅读书籍。', 'expected_mask': '<mask>', 'expected_output_contains': '图书馆'},
        {'input': '您好，<mask>一个苹果。', 'expected_mask': '<mask>', 'expected_output_contains': '给我'}
    ]

    for i, test_case in enumerate(test_cases):
        case_number = i + 1
        print(f"Testing case [{case_number}/{len(test_cases)}] started.")
        try:
            output = predict_masked_token(test_case['input'])
            assert test_case['expected_mask'] in test_case['input'], \
                f"Test case [{case_number}/{len(test_cases)}] failed: '<mask>' not found in input."
            assert test_case['expected_output_contains'] in output, \
                f"Test case [{case_number}/{len(test_cases)}] failed: Expected output to contain '{test_case['expected_output_contains']}', got '{output}'."
        except ValueError as e:
            assert str(e) == "The input text must contain exactly one masked token.", \
                f"Test case [{case_number}/{len(test_cases)}] failed: {e}"
    
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_masked_token()