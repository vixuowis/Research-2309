def test_fill_mask_chinese():
    """
    This function tests the fill_mask_chinese function with some example sentences.
    """
    test_sentence1 = '上海是中国的[MASK]大城市。'
    test_sentence2 = '北京是中国的[MASK]。'
    assert '最' in fill_mask_chinese(test_sentence1)
    assert '首都' in fill_mask_chinese(test_sentence2)

test_fill_mask_chinese()