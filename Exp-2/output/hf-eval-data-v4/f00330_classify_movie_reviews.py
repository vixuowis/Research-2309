# requirements_file --------------------

!pip install -U joblib,pandas

# function_import --------------------

import joblib
import pandas as pd

# function_code --------------------

def classify_movie_reviews(data_path):
    """
    This function loads a pre-trained movie review classification model and
    predicts the sentiment for each review in the given dataset.

    Parameters:
    data_path (str): The file path to the CSV file containing the movie reviews.

    Returns:
    list: A list of predicted sentiments for the reviews.
    """
    # Load the pre-trained model
    model = joblib.load('model.joblib')
    # Load the dataset
    data = pd.read_csv(data_path)
    # Make sure features in the data are aligned with the features in the model's config
    predictions = model.predict(data)
    return predictions.tolist()

# test_function_code --------------------

def test_classify_movie_reviews():
    print("Testing started.")
    # Prepare a sample dataset with expected format
    sample_data = {'review': ['Great movie!', 'Terrible movie!']}
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_data.csv', index=False)

    # Test if the function correctly predicts sentiments
    predictions = classify_movie_reviews('sample_data.csv')
    assert len(predictions) == 2, "Incorrect number of predictions received."
    print("Testing finished.")

# Run the test function
test_classify_movie_reviews()