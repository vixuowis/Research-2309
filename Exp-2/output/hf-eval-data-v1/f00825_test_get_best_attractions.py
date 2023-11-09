def test_get_best_attractions():
    """
    This function tests the get_best_attractions function by using a sample question.
    """
    question = 'What are the best attractions in Paris?'
    result = get_best_attractions(question)
    assert result.shape == (1, 768), 'The output shape is not correct'

test_get_best_attractions()