from f00035_tokenize_dataset import *
def test_tokenize_dataset():
    dataset = {"text": "This is a sample text."}
    assert tokenize_dataset(dataset) == ["This", "is", "a", "sample", "text."]
    dataset = {"text": "Another example."}
    assert tokenize_dataset(dataset) == ["Another", "example."]
    dataset = {"text": "Yet another example."}
    assert tokenize_dataset(dataset) == ["Yet", "another", "example."]
    print("All test cases pass.")

test_tokenize_dataset()
