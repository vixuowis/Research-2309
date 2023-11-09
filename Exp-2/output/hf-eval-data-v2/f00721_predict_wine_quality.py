# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd

# function_code --------------------

def predict_wine_quality():
    '''
    This function predicts the quality of wine based on its chemical properties.
    It uses a pre-trained model from the Hugging Face hub and the wine quality dataset.
    
    Args:
        None
    
    Returns:
        labels (array): The predicted labels for the wine samples.
        model_score (float): The accuracy score of the model on the dataset.
    
    Raises:
        Exception: If there is an error in loading the model or the dataset.
    '''
    REPO_ID = 'julien-c/wine-quality'
    FILENAME = 'sklearn_model.joblib'
    data_filename = 'winequality-red.csv'

    try:
        model = joblib.load(cached_download(hf_hub_url(REPO_ID, FILENAME)))
        data_file = cached_download(hf_hub_url(REPO_ID, data_filename))
    except Exception as e:
        print(f'Error in loading model or dataset: {e}')
        return

    wine_df = pd.read_csv(data_file, sep=';')
    X = wine_df.drop(['quality'], axis=1)
    Y = wine_df['quality']

    labels = model.predict(X)
    model_score = model.score(X, Y)

    return labels, model_score

# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function tests the predict_wine_quality function.
    It asserts that the returned labels and model score are not None.
    '''
    labels, model_score = predict_wine_quality()
    assert labels is not None, 'No labels returned'
    assert model_score is not None, 'No model score returned'
    assert 0 <= model_score <= 1, 'Invalid model score'

# call_test_function_code --------------------

test_predict_wine_quality()