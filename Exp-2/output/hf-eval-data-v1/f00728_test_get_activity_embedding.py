def test_get_activity_embedding():
    """
    This function tests the get_activity_embedding function.
    It loads the CortexBench dataset, selects a sample image, and checks if the output of the function is a tensor.
    """
    # Load the CortexBench dataset
    dataset = load_dataset('CortexBench')
    
    # Select a sample image
    img = dataset[0]['image']
    
    # Get the embedding of the image
    embedding = get_activity_embedding(img)
    
    # Check if the output is a tensor
    assert isinstance(embedding, torch.Tensor), 'The output should be a tensor.'
    
    # Check if the size of the tensor is correct
    assert embedding.size() == (1, embd_size), 'The size of the tensor should be (1, embd_size).'

test_get_activity_embedding()