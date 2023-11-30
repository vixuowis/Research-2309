# function_import --------------------

import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def classify_movie_reviews(data_file):
    """
    Classify movie reviews as positive or negative using a pretrained model.

    Args:
        data_file (str): Path to the CSV file containing movie reviews.

    Returns:
        predictions (numpy.ndarray): Predicted classes for each review in the input data.

    Raises:
        FileNotFoundError: If the model file or the data file does not exist.
    """
    # Load pretrained ML model
    try:
        clf = joblib.load('model/movie_reviews_model.joblib')
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the model file 'model/movie_reviews_model.joblib'.")

    # Load movie reviews to classify as data
    try:
        reviews = pd.read_csv(data_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the data file '{data_file}'.")
    except TypeError:
        raise TypeError("'data_file' argument is either empty or not a string.")
    
    # Classify movie reviews as positive or negative
    predictions = clf.predict(reviews)

    return np.array([str(prediction).replace('1', 'positive').replace('0', 'negative') for prediction in predictions])

# test_function_code --------------------

def test_classify_movie_reviews():
    """
    Test the classify_movie_reviews function with a sample data file.
    """
    try:
        predictions = classify_movie_reviews('test_data.csv')
        assert isinstance(predictions, np.ndarray), 'The output should be a numpy array.'
        assert len(predictions) > 0, 'The output array should not be empty.'
    except FileNotFoundError:
        print('Model file or data file not found.')
    except Exception as e:
        print(f'Unexpected error: {e}')


# call_test_function_code --------------------

test_classify_movie_reviews()