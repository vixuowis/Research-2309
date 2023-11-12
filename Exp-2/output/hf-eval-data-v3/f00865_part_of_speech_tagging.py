# function_import --------------------

from transformers import BertTokenizerFast, AutoModel

# function_code --------------------

def part_of_speech_tagging(chinese_sentence: str):
    """
    This function takes a Chinese sentence as input and returns the part-of-speech tags for each word in the sentence.
    
    Args:
        chinese_sentence (str): The Chinese sentence to be tagged.
    
    Returns:
        List[str]: A list of part-of-speech tags for each word in the sentence.
    
    Raises:
        OSError: If there is a problem loading the pretrained model or tokenizer.
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
    This function tests the part_of_speech_tagging function with a few test cases.
    """
    test_sentence_1 = '我爱北京天安门'
    test_sentence_2 = '今天天气真好'
    test_sentence_3 = '学习使我快乐'
    assert len(part_of_speech_tagging(test_sentence_1)) == len(test_sentence_1)
    assert len(part_of_speech_tagging(test_sentence_2)) == len(test_sentence_2)
    assert len(part_of_speech_tagging(test_sentence_3)) == len(test_sentence_3)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_part_of_speech_tagging()