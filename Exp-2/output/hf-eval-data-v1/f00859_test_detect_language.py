def test_detect_language():
    """
    Tests the detect_language function.
    """
    # Test with English text
    result = detect_language('Hello, how are you?')
    assert isinstance(result, list) and isinstance(result[0], dict)
    assert 'label' in result[0] and 'score' in result[0]

    # Test with French text
    result = detect_language('Bonjour, comment ça va?')
    assert isinstance(result, list) and isinstance(result[0], dict)
    assert 'label' in result[0] and 'score' in result[0]

    # Test with invalid input
    try:
        detect_language(123)
    except ValueError:
        pass
    else:
        assert False, 'Expected a ValueError.'

    print('All tests passed.')

test_detect_language()