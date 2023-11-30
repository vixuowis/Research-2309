# function_import --------------------

import joblib
import pandas as pd
import json
import numpy as np

# function_code --------------------

def calculate_carbon_emissions(data_file):
    """
    Calculate the carbon emissions for given data.

    Args:
        data_file (str): The path to the input data file in CSV format.

    Returns:
        numpy.ndarray: The predicted carbon emissions.

    Raises:
        FileNotFoundError: If the model or config file does not exist.
        pd.errors.EmptyDataError: If the data file is empty.
    """
    
    # load the trained model
    model = joblib.load('data/model.pkl')
    
    # load the feature mappings
    feature_mappings = json.loads(open("data/feature_names.json", "r").read()) 
    
    # read the input data into a pandas dataframe
    df = pd.read_csv(data_file)
    
    if df.empty:
        raise pd.errors.EmptyDataError('Input file is empty')
        
    features = [x for x in list(df)]
    
    # check the feature names to see whether they match with those used
    # during training, raise an exception otherwise
    
    if len(features) != len(feature_mappings):
        raise Exception('Mismatch between number of features')
        
    for x in range(len(features)):
        if feature_mappings[x]['feature'] != features[x]:
            raise Exception('Feature name mismatch: {} vs. {}'.format(\
                            feature_mappings[x]['feature'], features[x]))
    
    # remove the index columns and drop rows with null values
    df = df.drop(columns=['Unnamed: 0'])
    df = df.fillna(0)
        
    # get the data in numpy format, ready for inference
    X = []
    for feature_set in list(df):
        X.append(list(df[feature_set].values))
    
    X = np.array(X).reshape((1, -1)).astype('float')
        
    # apply the same preprocessing steps as during training
    X = (X - model['mean']) / model['std']
    
    return model['scaler'].transform(model['pipeline'].predict(X))

# test_function_code --------------------

def test_calculate_carbon_emissions():
    """Test the calculate_carbon_emissions function."""
    data_file = 'test_data.csv'
    try:
        predictions = calculate_carbon_emissions(data_file)
        assert isinstance(predictions, np.ndarray), 'The result should be a numpy array.'
        assert predictions.shape[0] > 0, 'The result should not be empty.'
    except FileNotFoundError:
        print('The model or config file does not exist.')
    except pd.errors.EmptyDataError:
        print('The data file is empty.')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_calculate_carbon_emissions()