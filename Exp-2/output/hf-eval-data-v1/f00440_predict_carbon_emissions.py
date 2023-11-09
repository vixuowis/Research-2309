import joblib
import pandas as pd

# Function to predict carbon emissions based on building features
# Parameters:
# - feat_x1, feat_x2, feat_x3: building features
# Returns:
# - Predicted carbon emissions category

def predict_carbon_emissions(feat_x1, feat_x2, feat_x3):
    # Load the model
    model = joblib.load('model.joblib')
    
    # Prepare the input data
    input_data = pd.DataFrame({"feat_x1": [feat_x1],
                               "feat_x2": [feat_x2],
                               "feat_x3": [feat_x3]})
    
    # Perform the prediction
    predictions = model.predict(input_data)
    
    return predictions[0]