# requirements_file --------------------

import subprocess

requirements = ["huggingface_hub", "joblib", "pandas", "pytest"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def recommend_wines_by_quality():
    """
    Recommends wines by loading a pre-trained skicit-learn model from the Hugging Face Hub,
    predicting the quality of wines in the dataset, and categorizing them based on quality.

    Args:
        None

    Returns:
        A tuple containing a DataFrame of wine features and the quality labels predicted by the model.

    Raises:
        IOError: If there is an issue loading the model or the dataset.
    """
    try:
        REPO_ID = "julien-c/wine-quality"
        FILENAME = "sklearn_model.joblib"

        model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
        data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))

        wine_df = pd.read_csv(data_file, sep=";")
        X = wine_df.drop(['quality'], axis=1)
        Y = wine_df['quality']

        labels = model.predict(X)
        return X, labels
    except Exception as e:
        raise IOError("Failed to load the model or dataset.") from e

# test_function_code --------------------

import pytest

from recommend_wines_by_quality import recommend_wines_by_quality

def test_recommend_wines_by_quality():
    print("Testing started.")

    print("Testing case [1/1] started.")
    features, labels = recommend_wines_by_quality()
    assert not features.empty and not labels.empty, "Test case [1/1] failed: No data returned."
    print("Testing finished.")

# call_test_function_line --------------------

test_recommend_wines_by_quality()