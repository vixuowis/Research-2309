# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def classify_movie_reviews(data_file):
    """
    Classify movie reviews as positive or negative using a pretrained model.

    Args:
        data_file (str): The path to the csv file containing movie reviews.

    Returns:
        predictions (list): A list of predictions where 1 represents a positive review and 0 represents a negative review.
    """
    model = joblib.load('model.joblib')
    data = pd.read_csv(data_file)
    predictions = model.predict(data)
    return predictions

# test_function_code --------------------

def test_classify_movie_reviews():
    """
    Test the function classify_movie_reviews.

    Raises:
        AssertionError: If the function does not work as expected.
    """
    predictions = classify_movie_reviews('test_data.csv')
    assert isinstance(predictions, list), 'The output should be a list.'
    assert all(isinstance(i, int) for i in predictions), 'All elements in the list should be integers.'
    assert all(i in [0, 1] for i in predictions), 'All elements in the list should be either 0 or 1.'

# call_test_function_code --------------------

test_classify_movie_reviews()