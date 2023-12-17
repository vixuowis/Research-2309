# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline, AutoModelForPreTraining, AutoTokenizer

# function_code --------------------

def fill_mask_portuguese(sentence: str) -> str:
    """
    Fills the masked token in a Portuguese sentence using the BERTimbau model.

    Args:
        sentence: A string containing a Portuguese sentence with a [MASK] token.

    Returns:
        A string with the [MASK] token filled with the most likely prediction.
    """

    model_name = 'neuralmind/bert-base-portuguese-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForPreTraining.from_pretrained(model_name)

    fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
    results = fill_mask(sentence)

    # Assuming there is only one [MASK] token in the sentence
    filled_sentence = results[0]['sequence']
    return filled_sentence


# test_function_code --------------------

def test_fill_mask_portuguese():
    print("Testing started.")

    # 测试用例 1：填充带单个 [MASK] 的句子
    print("Testing case [1/3] started.")
    sentence_with_mask = "Tinha uma [MASK] no meio do caminho."
    expected_result = "Tinha uma pedra no meio do caminho."  # Assuming 'pedra' is the predicted word
    assert fill_mask_portuguese(sentence_with_mask) == expected_result, f"Test case [1/3] failed: Expected {expected_result}"

    # 测试用例 2：填充带单个 [MASK] 的句子 (不同上下文)
    print("Testing case [2/3] started.")
    sentence_with_mask = "O gato estava [MASK] no sofá."
    expected_result = "O gato estava dormindo no sofá."  # Assuming 'dormindo' is the predicted word
    assert fill_mask_portuguese(sentence_with_mask) == expected_result, f"Test case [2/3] failed: Expected {expected_result}"

    # 测试用例 3：填充带多个 [MASK] 的句子
    print("Testing case [3/3] started.")
    sentence_with_mask = "Ele foi ao [MASK] comprar [MASK]."
    # Please note: This is a simplification, as the function is presumed to handle only one mask.
    # In a robust test, handling multiple masks should be tested properly.
    expected_result = "Ele foi ao mercado comprar pão."  # Assuming 'mercado' and 'pão' are the predicted words
    assert fill_mask_portuguese(sentence_with_mask) == expected_result, f"Test case [3/3] failed: Expected {expected_result}"
    print("Testing finished.")

# 运行测试功能
test_fill_mask_portuguese()
