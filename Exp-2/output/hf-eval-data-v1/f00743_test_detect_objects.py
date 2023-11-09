def test_detect_objects():
    # Test image URL
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'

    # Call the function with the test image URL
    logits, bboxes = detect_objects(url)

    # Check if the function returns outputs
    assert logits is not None, 'No objects detected'
    assert bboxes is not None, 'No bounding boxes detected'

    # Check if the function returns the correct types
    assert isinstance(logits, type(torch.Tensor())), 'Incorrect type for logits'
    assert isinstance(bboxes, type(torch.Tensor())), 'Incorrect type for bounding boxes'

    # Check if the function returns the correct shapes
    assert logits.shape[0] == bboxes.shape[0], 'Mismatch in number of objects and bounding boxes'

    # Call the function again with a different test image URL
    url = 'http://images.cocodataset.org/val2017/000000039770.jpg'
    logits, bboxes = detect_objects(url)

    # Check if the function returns outputs
    assert logits is not None, 'No objects detected'
    assert bboxes is not None, 'No bounding boxes detected'

    # Check if the function returns the correct types
    assert isinstance(logits, type(torch.Tensor())), 'Incorrect type for logits'
    assert isinstance(bboxes, type(torch.Tensor())), 'Incorrect type for bounding boxes'

    # Check if the function returns the correct shapes
    assert logits.shape[0] == bboxes.shape[0], 'Mismatch in number of objects and bounding boxes'