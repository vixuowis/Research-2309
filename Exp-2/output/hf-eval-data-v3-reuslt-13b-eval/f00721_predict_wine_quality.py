# function_import --------------------

from huggingface_hub import hf_hub_url, cached_download
import joblib
import pandas as pd
import numpy as np

# function_code --------------------

def predict_wine_quality():
    '''
    This function is used to predict the quality of wine based on its chemical properties.
    It uses a pre-trained model hosted on Hugging Face hub.
    
    Returns:
        tuple: A tuple containing the predicted labels and the model's score.
    
    Raises:
        Exception: If there is an error in loading the model or the data.
    '''
    # Get the data from user input
    try:
        # get data from request object 
        alcohol = float(request.form['alcohol'])
        volatile_acidity = float(request.form['volatile acidity'])
        sulphates = float(request.form['sulphates'])
        total_soluble_solids = float(request.form['total soluble solids'])
    except:
        # get data from url parameters 
        alcohol = request.args.get('alcohol')
        volatile_acidity = request.args.get('volatile acidity')
        sulphates = request.args.get('sulphates')
        total_soluble_solids = request.args.get('total soluble solids')
    alcohol = np.array(alcohol).reshape(-1, 1)
    volatile_acidity = np.array(volatile_acidity).reshape(-1, 1)
    sulphates = np.array(sulphates).reshape(-1, 1)
    total_soluble_solids = np.array(total_soluble_solids).reshape(-1, 1)
    
    # Load the model from Hugging Face hub.
    model_url=hf_hub_url(repo_id="nishantkr97/wine-quality", filename="wine_model.joblib")
    model = joblib.load(cached_download(model_url))
    
    # Create dataframe from the inputs
    data = pd.DataFrame([alcohol, volatile_acidity, sulphates, total_soluble_solids])
    print('Dataframe: ',data)
    
    # Get the prediction score and label using the model
    score = model.predict(data)[0]
    if score < 5:
        label="Bad"
    elif score > 5:
        label="Good"
        
    return (label, score)

# test_function_code --------------------

def test_predict_wine_quality():
    '''
    This function is used to test the predict_wine_quality function.
    It checks if the function returns the correct output type and if the model score is within an acceptable range.
    '''
    labels, score = predict_wine_quality()
    assert isinstance(labels, np.ndarray), 'The predicted labels should be a numpy array.'
    assert isinstance(score, float), 'The model score should be a float.'
    assert 0 <= score <= 1, 'The model score should be between 0 and 1.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_predict_wine_quality()