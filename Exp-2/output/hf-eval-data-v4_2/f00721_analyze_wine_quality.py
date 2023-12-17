# requirements_file --------------------

!pip install -U huggingface_hub joblib pandas

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def analyze_wine_quality(repo_id: str, model_filename: str, data_filename: str) -> float:
    """
    Analyzes the wine quality based on chemical properties using a pre-trained model.

    Args:
        repo_id: The Hugging Face repository ID containing the model and dataset.
        model_filename: The name of the file containing the pre-trained model.
        data_filename: The name of the CSV file containing the wine quality dataset.

    Returns:
        The accuracy score of the pre-trained model on the dataset.

    Raises:
        FileNotFoundError: If the provided files are not found.
    """
    model = joblib.load(cached_download(hf_hub_url(repo_id, model_filename)))
    data_file = cached_download(hf_hub_url(repo_id, data_filename))
    
    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']

    labels = model.predict(X)
    return model.score(X, Y)

# test_function_code --------------------

def test_analyze_wine_quality():
    print("Testing started.")
    
    # Set up the test data and model information
    repo_id = 'julien-c/wine-quality'
    model_filename = 'sklearn_model.joblib'
    data_filename = 'winequality-red.csv'

    # Test case 1: Check if function returns a float
    print("Testing case [1/1] started.")
    accuracy = analyze_wine_quality(repo_id, model_filename, data_filename)
    assert isinstance(accuracy, float), f"Test case [1/1] failed: Expected float, got {type(accuracy)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_wine_quality()