def test_fill_mask_with_legal_bert():
    # Test the function with some example sentences
    test_sentence1 = 'The defendant is [MASK] guilty of the crime.'
    test_sentence2 = 'The contract is [MASK] binding.'
    test_sentence3 = 'The law states that [MASK] shall not discriminate.'

    # Call the function with the test sentences
    print(fill_mask_with_legal_bert(test_sentence1))
    print(fill_mask_with_legal_bert(test_sentence2))
    print(fill_mask_with_legal_bert(test_sentence3))

    # Assert that the function returns a string
    assert isinstance(fill_mask_with_legal_bert(test_sentence1), str)
    assert isinstance(fill_mask_with_legal_bert(test_sentence2), str)
    assert isinstance(fill_mask_with_legal_bert(test_sentence3), str)

test_fill_mask_with_legal_bert()