from f00376_preprocess_function import *
def test_preprocess_function():
    examples = {
        "text": [
            "This is the first document.",
            "This document is the second document.",
            "And this is the third one.",
            "Is this the first document?",
            "The last document?"
        ],
        "summary": [
            "This is the first summary.",
            "This is the second summary.",
            "This is the third summary.",
            "This is the fourth summary.",
            "This is the fifth summary."
        ]
    }
    expected_result = {
        "input_ids": [[0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2]],
        "attention_mask": [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
        "labels": [[0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2], [0, 3, 9, 5, 7, 2]]
    }
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    result = preprocess_function(examples)
    assert result == expected_result, f"Expected {expected_result}, but got {result}"

test_preprocess_function()
