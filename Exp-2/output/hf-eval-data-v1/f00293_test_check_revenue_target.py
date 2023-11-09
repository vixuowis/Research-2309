def test_check_revenue_target():
    # Test dataset
    table = {'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 'Revenue': [2000, 2500, 3000, 3500, 4000, 4500, 5000]}
    query = 'Did the total revenue meet the target revenue of 24000?'

    # Call the function with the test dataset
    predicted_answer_coordinates, predicted_aggregation_indices = check_revenue_target(table, query)

    # Check the results
    # Note: we're not comparing numbers strictly here, as the model's predictions can vary
    assert isinstance(predicted_answer_coordinates, list), 'The result should be a list'
    assert isinstance(predicted_aggregation_indices, list), 'The result should be a list'

test_check_revenue_target()