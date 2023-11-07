from f00840_tokenize_text import *
def test_tokenize_text():
	text = "Don't you love 🤗 Transformers? We sure do."
	expected_tokens = ["▁Don", "'", "t", "▁you", "▁love", "▁", "🤗", "▁", "Transform", "ers", "?", "▁We", "▁sure", "▁do", "."]
	tokens = tokenize_text(text)
	assert tokens == expected_tokens

# Additional test cases
# test case 1
# text = "I have a new GPU"
# expected_tokens = ["I", "have", "a", "new", "GP", "##U"]
# tokens = tokenize_text(text)
# assert tokens == expected_tokens

# test case 2
# text = "Hello, world!"
# expected_tokens = ["Hello", ",", "▁world", "!"]
# tokens = tokenize_text(text)
# assert tokens == expected_tokens

# test case 3
# text = "This is a test."
# expected_tokens = ["This", "▁is", "▁a", "▁test", "."]
# tokens = tokenize_text(text)
# assert tokens == expected_tokens

# test case 4
# text = "I love coding."
# expected_tokens = ["I", "▁love", "▁coding", "."]
# tokens = tokenize_text(text)
# assert tokens == expected_tokens

# test case 5
# text = "I am happy."
# expected_tokens = ["I", "▁am", "▁happy", "."]
# tokens = tokenize_text(text)
# assert tokens == expected_tokens

test_tokenize_text()
