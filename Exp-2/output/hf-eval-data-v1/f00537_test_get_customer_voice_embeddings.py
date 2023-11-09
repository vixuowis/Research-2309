def test_get_customer_voice_embeddings():
    '''
    This function tests the get_customer_voice_embeddings function.
    It uses a sample audio file for testing.
    '''
    # Path to the sample audio file
    audio_file = 'tests/samples/ASR/spk1_snt1.wav'
    
    # Get the voice embeddings of the customer
    embeddings = get_customer_voice_embeddings(audio_file)
    
    # Check if the embeddings are not None
    assert embeddings is not None, 'The embeddings should not be None.'
    
    # Check if the embeddings are of the correct type
    assert isinstance(embeddings, torch.Tensor), 'The embeddings should be a torch.Tensor.'
    
    # Check if the embeddings have the correct shape
    assert embeddings.shape[0] > 0, 'The embeddings should have a shape of (N, D) where N > 0 and D is the dimension of the embeddings.'

test_get_customer_voice_embeddings()