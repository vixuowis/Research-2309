from f00111_tokenize_function import *
def test_tokenize_function():
    examples = {
        "text": [
            "This is the first example.",
            "This is the second example."
        ]
    }
    tokenized_examples = tokenize_function(examples)
    assert len(tokenized_examples["input_ids"]) == 2
    assert len(tokenized_examples["attention_mask"]) == 2
    assert len(tokenized_examples["token_type_ids"]) == 2
    assert len(tokenized_examples["text"]) == 2
    print("All test cases pass.")

test_tokenize_function()
