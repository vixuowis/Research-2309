def test_calculate_review_similarity():
    # Define a small set of restaurant reviews for testing
    test_reviews = ['The food was delicious.', 'I loved the food.', 'The food was not good.']
    
    # Call the function with the test reviews
    similarity_matrix = calculate_review_similarity(test_reviews)
    
    # Since we're dealing with floating point numbers, we can't compare them strictly.
    # Instead, we'll check if the similarity scores are within a reasonable range (0 to 1).
    for row in similarity_matrix:
        for score in row:
            assert 0 <= score <= 1, 'Invalid similarity score'
    
    print('All tests passed.')

# Run the test function
test_calculate_review_similarity()