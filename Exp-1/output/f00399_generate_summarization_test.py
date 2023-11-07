from f00399_generate_summarization import *
def test_generate_summarization():
    inputs = "This is a test input."
    expected_output = "This is a test output."
    assert generate_summarization(inputs) == expected_output

    inputs = "Another test input."
    expected_output = "Another test output."
    assert generate_summarization(inputs) == expected_output

    inputs = "Yet another test input."
    expected_output = "Yet another test output."
    assert generate_summarization(inputs) == expected_output
