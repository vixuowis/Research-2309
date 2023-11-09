def test_summarize_text():
    # Test text
    text = "The quick brown fox jumps over the lazy dog. The dog was not amused."
    # Expected output
    expected_output = "A quick brown fox jumps over a lazy dog, who is not amused."
    # Call the summarize_text function
    output = summarize_text(text)
    # Assert that the output is as expected
    assert output in expected_output, f'Expected {expected_output}, but got {output}'
    print('Test passed.')

test_summarize_text()