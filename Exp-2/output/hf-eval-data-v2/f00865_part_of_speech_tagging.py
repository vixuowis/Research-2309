# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def part_of_speech_tagging(chinese_sentence):
    """
    This function performs part-of-speech tagging on a given Chinese sentence.

    Args:
        chinese_sentence (str): The Chinese sentence to be tagged.

    Returns:
        part_of_speech_tags (torch.Tensor): The part-of-speech tags for all tokens in the sentence.
    """
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
    model = AutoModel.from_pretrained('ckiplab/bert-base-chinese-pos')

    tokens = tokenizer(chinese_sentence, return_tensors='pt')
    outputs = model(**tokens)
    part_of_speech_tags = outputs.logits.argmax(-1)

    return part_of_speech_tags

# test_function_code --------------------

def test_part_of_speech_tagging():
    """
    This function tests the part_of_speech_tagging function.
    """
    chinese_sentence = '我爱北京天安门'
    pos_tags = part_of_speech_tagging(chinese_sentence)

    assert pos_tags is not None, 'The function should return a value.'
    assert pos_tags.size(0) == len(chinese_sentence), 'The number of tags should be equal to the number of tokens in the sentence.'

# call_test_function_code --------------------

test_part_of_speech_tagging()