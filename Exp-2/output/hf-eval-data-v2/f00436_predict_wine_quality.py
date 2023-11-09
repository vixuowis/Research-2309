# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def predict_wine_quality():
    '''
    This function loads a pre-trained Scikit-learn model from the Hugging Face Hub and uses it to predict wine quality.
    The model and dataset are hosted on the Hugging Face Hub under the repository 'julien-c/wine-quality'.
    The function loads the model using joblib and the dataset using pandas.
    The dataset is split into features (X) and target labels (Y).
    The function returns the predicted wine quality labels for the input data.
    
    Returns:
    labels: array, shape = [n_samples]
        Predicted wine quality labels for each sample in X.
    '''
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'
    
    model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
    data_file = cached_download(hf_hub_url(REPO_ID, 'winequality-red.csv'))
    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    
    labels = model.predict(X)
    return labels

# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function tests the predict_wine_quality function by loading a sample of the dataset and comparing the predicted labels with the actual labels.
    The test passes if the predicted labels match the actual labels for the majority of the samples.
    '''
    labels = predict_wine_quality()
    data_file = cached_download(hf_hub_url('julien-c/wine-quality', 'winequality-red.csv'))
    wine_df = pd.read_csv(data_file, sep=';')
    Y = wine_df['quality']
    
    assert (labels == Y).mean() > 0.5, 'Test failed: The predicted labels do not match the actual labels for the majority of the samples.'

# call_test_function_code --------------------

test_predict_wine_quality()