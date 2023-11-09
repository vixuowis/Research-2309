def test_chinese_pos_tagging():
    """
    This function tests the chinese_pos_tagging function.
    """
    test_sentence = '我爱吃苹果。'
    pos_tags = chinese_pos_tagging(test_sentence)
    assert pos_tags is not None, 'The function should return a value.'
    assert pos_tags.shape[0] == len(test_sentence.split()), 'The number of tags should be equal to the number of words in the sentence.'

test_chinese_pos_tagging()