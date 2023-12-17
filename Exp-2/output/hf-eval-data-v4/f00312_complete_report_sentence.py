# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def complete_report_sentence(masked_sentence: str) -> str:
    """
    This function takes a sentence with a mask token and uses the XLM-RoBERTa model 
    to predict the missing part of the sentence.

    Parameters:
    - masked_sentence (str): A sentence containing a mask token ('<mask>') where the 
                             AI model should fill in the blank.

    Returns:
    - str: The sentence completed by the AI model.
    """
    unmasker = pipeline('fill-mask', model='xlm-roberta-base')
    # Perform the completion
    completed_results = unmasker(masked_sentence)
    # Extract the top result (highest score)
    completed_sentence = completed_results[0]['sequence']
    return completed_sentence

# test_function_code --------------------

def test_complete_report_sentence():
    print("Testing started.")
    
    # 测试用例 1: 一个控码标记的简单语句
    print("Testing case [1/3] started.")
    sentence_1 = "During the meeting, we discussed the <mask> for the next quarter."
    result_1 = complete_report_sentence(sentence_1)
    assert "<mask>" not in result_1, f"Test case [1/3] failed: Mask not replaced in the sentence."

    # 测试用例 2: 包含多个控码标记，但仅填写第一个
    print("Testing case [2/3] started.")
    sentence_2 = "The quarterly results exceeded <mask> expectations, leading to an increase in <mask>."
    result_2 = complete_report_sentence(sentence_2)
    assert "<mask>" in result_2.split(' ', 1)[1], f"Test case [2/3] failed: Multiple masks detected; only first should be replaced."

    # 测试用例 3: 句子中不包含控码标记
    print("Testing case [3/3] started.")
    sentence_3 = "We achieved a significant milestone this year."
    result_3 = complete_report_sentence(sentence_3)
    assert result_3 == sentence_3, f"Test case [3/3] failed: Sentence without mask should remain unchanged."

    print("Testing finished.")

# 运行测试函数
test_complete_report_sentence()