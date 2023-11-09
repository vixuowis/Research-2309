def test_classify_image_content():
    # Test the function with a sample image URL
    image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    probs = classify_image_content(image_url)

    # Check if the function returns a list
    assert isinstance(probs, list), 'The function should return a list.'

    # Check if the list contains two elements (for the two labels: cat and dog)
    assert len(probs) == 2, 'The list should contain two elements.'

    # Check if the sum of the probabilities is approximately 1 (because they are probabilities)
    assert abs(sum(probs) - 1) < 1e-6, 'The sum of the probabilities should be approximately 1.'

test_classify_image_content()