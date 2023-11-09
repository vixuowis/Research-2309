def test_classify_plant_disease():
    """
    This function tests the classify_plant_disease function.
    It uses an online image of a healthy plant for the test.
    """
    # Define the path to the test image
    test_image_path = 'https://example.com/healthy_plant.jpg'
    
    # Call the function with the test image
    classification_results = classify_plant_disease(test_image_path)
    
    # Check the type of the result
    assert isinstance(classification_results, dict), 'The result should be a dictionary.'
    
    # Check the keys of the result
    expected_labels = ['healthy', 'pest-infested', 'fungus-infected', 'nutrient-deficient']
    assert set(classification_results.keys()) == set(expected_labels), 'The result should contain the expected labels.'
    
    # Check the values of the result
    for label, prob in classification_results.items():
        assert isinstance(prob, float), 'The probabilities should be floats.'
        assert 0 <= prob <= 1, 'The probabilities should be between 0 and 1.'