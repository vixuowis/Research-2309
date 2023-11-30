# function_import --------------------

import json
import joblib
import pandas as pd

# function_code --------------------

def predict_carbon_emissions(model_path: str, config_path: str, data_path: str) -> pd.DataFrame:
    """
    Predict the carbon emissions of different facilities based on the provided data.

    Args:
        model_path (str): The path to the pretrained model.
        config_path (str): The path to the configuration file.
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The predicted carbon emissions for each facility.
    """    
    
    # Load model and configuration file 
    config = json.load(open(config_path))

    # load data
    df = pd.read_csv(data_path)

    df['facility']=df['Facility Name'].astype('category').cat.codes
    
    df['month'] = pd.Categorical(pd.to_datetime(df['date']).dt.strftime("%b"), categories=config['categories'])
        .cat.codes
    df.head()
    
    # Predict carbon emissions using model
    with open(model_path, 'rb') as mdl:
        
        model = joblib.load(mdl)
            
        predicted_values = model.predict(df[list(config['x'])])
    
    return pd.DataFrame({'Facility Name': df['Facility Name'],  'CO2_predicted': predicted_values})

# test_function_code --------------------

def test_predict_carbon_emissions():
    """
    Test the predict_carbon_emissions function.
    """
    model_path = 'model.joblib'
    config_path = 'config.json'
    data_path = 'data.csv'

    try:
        predictions = predict_carbon_emissions(model_path, config_path, data_path)
        assert isinstance(predictions, pd.DataFrame), 'The result is not a DataFrame.'
        assert not predictions.empty, 'The DataFrame is empty.'
    except FileNotFoundError:
        print('Test files not found.')
    except Exception as e:
        print(f'An error occurred: {e}')
    else:
        print('All tests passed.')


# call_test_function_code --------------------

test_predict_carbon_emissions()