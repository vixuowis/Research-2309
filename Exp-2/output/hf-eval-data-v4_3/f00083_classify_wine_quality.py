# requirements_file --------------------

import subprocess

requirements = ["huggingface_hub", "joblib", "pandas"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def classify_wine_quality(data_file: str) -> pd.DataFrame:
    """
    Classifies the quality of wine based on given features.

    Args:
        data_file: A string path to the CSV file containing wine features.

    Returns:
        A DataFrame with the original features and a new 'predicted_quality' column.

    Raises:
        FileNotFoundError: If the data_file does not exist.
        Exception: If the model fails to predict.
    """
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'

    # Load the pre-trained model
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))

    # Load the wine features from CSV
    winedf = pd.read_csv(data_file, sep=';')

    # Prepare the feature matrix X by dropping the 'quality' column if it exists
    X = winedf.drop(['quality'], axis=1, errors='ignore')

    # Make predictions
    predictions = model.predict(X)

    # Add predictions to the DataFrame
    winedf['predicted_quality'] = predictions
    return winedf


# test_function_code --------------------

def test_classify_wine_quality():
    print("Testing started.")
    sample_data = cached_download(hf_hub_url('julien-c/wine-quality', 'sample-winequality-red.csv'))

    # Test case 1: Verify that the function returns a DataFrame with the correct number of columns
    print("Testing case [1/3] started.")
    result_df = classify_wine_quality(sample_data)
    assert 'predicted_quality' in result_df.columns, "Test case [1/3] failed: 'predicted_quality' column not found."

    # Test case 2: Verify that the function adds exactly one column to the input DataFrame
    print("Testing case [2/3] started.")
    expected_column_count = pd.read_csv(sample_data, sep=';').shape[1] + 1
    assert result_df.shape[1] == expected_column_count, f"Test case [2/3] failed: Expected {expected_column_count} columns but found {result_df.shape[1]}"

    # Test case 3: Verify that the predicted_quality column is non-null
    print("Testing case [3/3] started.")
    assert result_df['predicted_quality'].isnull().sum() == 0, "Test case [3/3] failed: 'predicted_quality' column contains null values."
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_wine_quality()