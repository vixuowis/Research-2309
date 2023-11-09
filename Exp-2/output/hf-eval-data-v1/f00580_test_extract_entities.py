def test_extract_entities():
    # Test the function with a sample sentence
    entities = extract_entities("Apple's CEO is Tim Cook and Microsoft's CEO is Satya Nadella")
    # Assert that the function returns a tensor (without checking the exact values)
    assert isinstance(entities, torch.Tensor)
    # Load the dataset used for performance evaluation
    dataset = torch.load('ismail-lucifer011/autotrain-data-name_all')
    # Select a sample from the dataset
    sample = dataset[0]
    # Test the function with the sample from the dataset
    entities = extract_entities(sample)
    # Assert that the function returns a tensor (without checking the exact values)
    assert isinstance(entities, torch.Tensor)

test_extract_entities()