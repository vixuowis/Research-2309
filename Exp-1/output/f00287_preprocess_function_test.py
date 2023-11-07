from f00287_preprocess_function import *
def test_preprocess_function():
    examples = {
        "answers.text": [
            ["This", "is", "the", "first", "example."],
            ["This", "is", "the", "second", "example."],
            ["This", "is", "the", "third", "example."]
        ]
    }
    expected_output = [
        ["This", "is", "the", "first", "example", "."],
        ["This", "is", "the", "second", "example", "."],
        ["This", "is", "the", "third", "example", "."]
    ]
    assert preprocess_function(examples) == expected_output

test_preprocess_function()
