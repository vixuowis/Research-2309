def test_detect_suspicious_objects():
    """
    Test the detect_suspicious_objects function.
    """
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    texts = ['a photo of a suspicious person', 'a photo of a suspicious object']
    results = detect_suspicious_objects(url, texts)
    assert isinstance(results, dict), 'The result should be a dictionary.'
    assert 'scores' in results, 'The result should contain scores.'
    assert 'labels' in results, 'The result should contain labels.'
    assert 'boxes' in results, 'The result should contain boxes.'