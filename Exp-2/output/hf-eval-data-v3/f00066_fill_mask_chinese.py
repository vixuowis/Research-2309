# function_import --------------------

from transformers import BertTokenizer, AlbertForMaskedLM, FillMaskPipeline

# function_code --------------------

def fill_mask_chinese(sentence: str) -> str:
    """
    Fill in the mask in a Chinese sentence using a pre-trained model.

    Args:
        sentence (str): The sentence with a [MASK] token to be filled.

    Returns:
        str: The sentence with the [MASK] token filled.
    """
    tokenizer = BertTokenizer.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    model = AlbertForMaskedLM.from_pretrained('uer/albert-base-chinese-cluecorpussmall')
    unmasker = FillMaskPipeline(model, tokenizer)
    filled_sentence = unmasker(sentence)[0]['sequence']
    return filled_sentence

# test_function_code --------------------

def test_fill_mask_chinese():
    assert fill_mask_chinese('上海是中国的[MASK]大城市。') == '上海是中国的最大城市。'
    assert fill_mask_chinese('北京是中国的[MASK]。') == '北京是中国的首都。'
    assert fill_mask_chinese('我最喜欢的水果是[MASK]。') == '我最喜欢的水果是苹果。'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_fill_mask_chinese()