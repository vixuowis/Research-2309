from f00695_tokenizer import *
def test_tokenizer():
    text = "This is a sample text."
    return_tensors = "pt"
    encoded_text = tokenizer(text, return_tensors)
    assert isinstance(encoded_text, torch.Tensor)

    text = "Another sample text."
    return_tensors = "pt"
    encoded_text = tokenizer(text, return_tensors)
    assert isinstance(encoded_text, torch.Tensor)

    text = "Yet another sample text."
    return_tensors = "pt"
    encoded_text = tokenizer(text, return_tensors)
    assert isinstance(encoded_text, torch.Tensor)

    text = "One more sample text."
    return_tensors = "pt"
    encoded_text = tokenizer(text, return_tensors)
    assert isinstance(encoded_text, torch.Tensor)

    text = "Final sample text."
    return_tensors = "pt"
    encoded_text = tokenizer(text, return_tensors)
    assert isinstance(encoded_text, torch.Tensor)
