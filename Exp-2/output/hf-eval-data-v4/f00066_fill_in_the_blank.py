# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BertTokenizer, AlbertForMaskedLM, FillMaskPipeline

# function_code --------------------

def fill_in_the_blank(text):
    """
    Fill in the blank in a given sentence with the most appropriate word using a pre-trained ALBERT model.

    Parameters:
    text (str): The sentence containing the [MASK] token to be filled.

    Returns:
    str: The sentence with the [MASK] token filled with the most appropriate word predicted by the model.
    """
    tokenizer = BertTokenizer.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    model = AlbertForMaskedLM.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    unmasker = FillMaskPipeline(model, tokenizer)
    result = unmasker(text)[0]['sequence']
    return result

# test_function_code --------------------

def test_fill_in_the_blank():
    print("Testing fill_in_the_blank function.")
    sentence = "上海是中国的[MASK]大城市。"
    expected_output = "上海是中国的第二大城市。"
    filled_sentence = fill_in_the_blank(sentence)
    assert filled_sentence == expected_output, f"Test failed: Expected {expected_output}, but got {filled_sentence}"
    print("Test passed successfully.")

# Running the test function
test_fill_in_the_blank()