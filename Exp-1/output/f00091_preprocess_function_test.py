from f00091_preprocess_function import *
def test_preprocess_function():
    examples = {
        "audio": [
            {"array": [1, 2, 3]},
            {"array": [4, 5, 6, 7]},
            {"array": [8, 9]},
        ]
    }
    expected_output = "..."  # Replace with expected output
    assert preprocess_function(examples) == expected_output

test_preprocess_function()
