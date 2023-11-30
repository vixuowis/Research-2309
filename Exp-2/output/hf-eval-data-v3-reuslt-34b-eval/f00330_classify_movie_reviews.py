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
    
    try:
        # load the model and vectorizer
        model = joblib.load("model.sav")
        vectorizer = joblib.load("vectorizer.sav")
        
        # read in data from csv
        df = pd.read_csv(data_file)
        
        # get the review column and vectorize it using the loaded vectorizer
        reviews = np.array(df["review"])
        X = vectorizer.transform(reviews).toarray()
    
        # get model predictions for all data in the csv file
        predictions = model.predict(X)
    except FileNotFoundError:
        raise
    
    return predictions


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