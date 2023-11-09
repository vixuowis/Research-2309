def test_fill_mask_dutch_sentence():
    """
    This function tests the fill_mask_dutch_sentence function by using a sample sentence.
    """
    test_sentence = "Het is vandaag erg koud, dus vergeet niet je ___ mee te nemen."
    expected_output = "Het is vandaag erg koud, dus vergeet niet je jas mee te nemen."
    assert fill_mask_dutch_sentence(test_sentence) == expected_output

test_fill_mask_dutch_sentence()