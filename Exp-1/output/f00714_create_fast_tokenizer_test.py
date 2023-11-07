from f00714_create_fast_tokenizer import *
def test_create_fast_tokenizer():
    fast_tokenizer = create_fast_tokenizer()

    # Test case 1
    text = "This is a test sentence."
    expected_tokens = ["this", "is", "a", "test", "sentence", "."]
    assert fast_tokenizer.tokenize(text) == expected_tokens

    # Test case 2
    text = "Another example sentence."
    expected_tokens = ["another", "example", "sentence", "."]
    assert fast_tokenizer.tokenize(text) == expected_tokens

    # Test case 3
    text = "Hello world!"
    expected_tokens = ["hello", "world", "!"]
    assert fast_tokenizer.tokenize(text) == expected_tokens

    # Test case 4
    text = "I love transformers library."
    expected_tokens = ["i", "love", "transformers", "library", "."]
    assert fast_tokenizer.tokenize(text) == expected_tokens

    # Test case 5
    text = "The quick brown fox jumps over the lazy dog."
    expected_tokens = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    assert fast_tokenizer.tokenize(text) == expected_tokens

test_create_fast_tokenizer()
