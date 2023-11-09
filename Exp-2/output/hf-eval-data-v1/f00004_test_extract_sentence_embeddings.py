def test_extract_sentence_embeddings():
    """
    This function tests the extract_sentence_embeddings function.
    It uses a sample sentence in English and checks if the output is a tensor.
    """
    # Define a sample sentence
    sample_sentence = 'Here is a sentence in English.'
    
    # Call the function with the sample sentence
    output = extract_sentence_embeddings(sample_sentence)
    
    # Check if the output is a tensor
    assert isinstance(output, torch.Tensor), 'Output is not a tensor.'
    
    # Check if the output has the correct shape
    assert output.shape == (1, 768), 'Output shape is incorrect.'

test_extract_sentence_embeddings()