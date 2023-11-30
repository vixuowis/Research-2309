# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_mask_chinese(text):
    """
    This function uses the 'bert-base-chinese' model to predict the most appropriate word to fill in the masked token in the Chinese text.

    Args:
        text (str): A string of text in Chinese with a masked token.

    Returns:
        list: A list of dictionaries with the predicted tokens and their corresponding scores.

    Raises:
        PipelineException: If no mask_token ([MASK]) is found on the input.
    """

    unmasker = pipeline('fill-mask', model='bert-base-chinese')
    return unmasker(text)


# test_function_code --------------------

def test_fill_mask_chinese():
    """
    This function tests the 'fill_mask_chinese' function with different test cases.
    """
    # Test case 1: Normal case with one masked token
    text1 = '我们很高兴与您合作，希望我们的<mask>能为您带来便利。'
    result1 = fill_mask_chinese(text1)
    assert isinstance(result1, list) and len(result1) > 0, 'Test case 1 failed'

    # Test case 2: Case with multiple masked tokens
    text2 = '我们很<mask>与您合作，希望我们的<mask>能为您带来便利。'
    try:
        result2 = fill_mask_chinese(text2)
    except Exception as e:
        assert str(e) == 'No mask_token ([MASK]) found on the input', 'Test case 2 failed'

    # Test case 3: Case with no masked tokens
    text3 = '我们很高兴与您合作，希望我们的产品能为您带来便利。'
    try:
        result3 = fill_mask_chinese(text3)
    except Exception as e:
        assert str(e) == 'No mask_token ([MASK]) found on the input', 'Test case 3 failed'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_fill_mask_chinese()