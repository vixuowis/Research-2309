def test_summarize_text():
    """
    This function tests the summarize_text function by using a sample text.
    """
    # Sample text
    text = "BigBird, a sparse-attention based transformer, extends Transformer-based models like BERT to much longer sequences. It can handle sequences up to a length of 4096 at a much lower compute cost compared to BERT. BigBird has achieved state-of-the-art results on various tasks involving very long sequences such as long documents summarization and question-answering with long contexts."
    
    # Expected output (this is a hypothetical output, the actual output may vary)
    expected_output = "BigBird is a sparse-attention based transformer that extends models like BERT to handle longer sequences up to a length of 4096. It achieves state-of-the-art results on tasks involving long sequences such as document summarization and question-answering."
    
    # Call the summarize_text function
    output = summarize_text(text)
    
    # Assert that the output is a string
    assert isinstance(output, str), f'Expected string, got {type(output)}'
    
    # Assert that the output is not empty
    assert len(output) > 0, 'Output is empty'
    
    # Assert that the output is not equal to the input text
    assert output != text, 'Output is equal to input text'
    
    # Assert that the output is shorter than the input text
    assert len(output) < len(text), 'Output is not shorter than input text'

test_summarize_text()