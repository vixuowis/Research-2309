from f00395_tokenize_text import *
def test_tokenize_text():
    text = "This is a sample text."
    expected_output = torch.tensor([[101, 2023, 2003, 1037, 7099, 3793, 1012, 102]])
    assert torch.equal(tokenize_text(text), expected_output)

test_tokenize_text()
