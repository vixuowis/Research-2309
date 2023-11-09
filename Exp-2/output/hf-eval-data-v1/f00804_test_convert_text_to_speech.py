def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function.
    """
    text = 'こんにちは、世界'
    output = convert_text_to_speech(text)
    assert isinstance(output, torch.Tensor), 'Output should be a torch.Tensor.'

test_convert_text_to_speech()