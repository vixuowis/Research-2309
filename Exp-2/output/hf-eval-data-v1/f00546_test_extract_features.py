def test_extract_features():
    # Test the extract_features function with a sample entity name
    entity_names = 'covid infection'
    cls_embedding = extract_features(entity_names)

    # Assert that the output is not None
    assert cls_embedding is not None

    # Assert that the output is a tensor
    assert isinstance(cls_embedding, torch.Tensor)

    # Assert that the size of the output tensor is as expected
    assert cls_embedding.size() == (1, 768)

test_extract_features()