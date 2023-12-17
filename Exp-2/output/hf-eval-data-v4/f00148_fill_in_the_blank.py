# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def fill_in_the_blank(sentence_with_mask):
    """
    This function uses a pre-trained DeBERTa model to fill in the blanks in a sentence.
    
    :param sentence_with_mask: A string with one instance of '[MASK]' representing the missing word.
    :return: A list of dictionaries with possible words to fill the blank along with their confidence scores.
    """
    fill_mask = pipeline('fill-mask', model='microsoft/deberta-v3-base')
    return fill_mask(sentence_with_mask)

# test_function_code --------------------

def test_fill_in_the_blank():
    print("Testing started.")
    test_sentence = "The artificial intelligence technology called '[MASK]' is advancing rapidly."

    # 测试用例 1：检查返回值是否为列表
    print("Testing case [1/3] started.")
    result = fill_in_the_blank(test_sentence)
    assert isinstance(result, list), f"Test case [1/3] failed: Expected result to be a list, but got {type(result)}."
    print("Testing case [1/3] finished.")

    # 测试用例 2：检查列表中的元素是否为字典
    print("Testing case [2/3] started.")
    assert all(isinstance(item, dict) for item in result), "Test case [2/3] failed: Expected all items in result to be dictionaries."
    print("Testing case [2/3] finished.")

    # 测试用例 3：检查每个字典中是否包含'sequence'和'score'键
    print("Testing case [3/3] started.")
    assert all("sequence" in item and "score" in item for item in result), "Test case [3/3] failed: Expected each dictionary to contain 'sequence' and 'score' keys."
    print("Testing case [3/3] finished.")
    
    print("Testing finished.")