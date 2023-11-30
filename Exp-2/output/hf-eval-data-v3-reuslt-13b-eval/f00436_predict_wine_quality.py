# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_wine_quality():
    '''
    This function loads a Scikit-learn model from the Hugging Face Hub and uses it to predict wine quality.
    
    Returns:
        labels (numpy.ndarray): The predicted quality labels for the wines.
    '''
    
    # Download model file
    url = hf_hub_url(repo_id="datascience-ai/mlflow-project", filename="best_model.joblib")
    cached_download(url, cache_dir=".cache", force_filename="best_model.joblib")
    
    # Load model and get predictor function
    loaded_model = joblib.load("best_model.joblib")
    predictor = loaded_model['artifacts']['sk_model']["model"]
    
    # Read data from csv file in the current working directory. This is a dummy dataset
    # that will be used for testing purposes only. In your own project, you would read data from
    # some external source (i.e. an API) into a DataFrame and then use this to predict values using the model.
    
    df_predict = pd.read_csv("data/wine_quality_test.csv")
    X_predict = df_predict.drop(columns=["quality_label"]).values
    
    # Predict wine quality
    predicted_labels = predictor.predict(X_predict)
    
    return predicted_labels

# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function tests the predict_wine_quality function by checking the shape and dtype of the output.
    '''
    labels = predict_wine_quality()
    assert isinstance(labels, np.ndarray), 'Output should be a numpy array.'
    assert labels.shape[0] == 1599, 'Output shape should be (1599,).'
    assert labels.dtype == np.int64, 'Output dtype should be int64.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_predict_wine_quality()