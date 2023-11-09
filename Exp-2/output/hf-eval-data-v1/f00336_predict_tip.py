from huggingface_hub import hf_hub_download
import joblib

# Function to predict the tip given by a new customer
# based on different input features like total bill, sex, smoker, day, time, and party size.
def predict_tip(total_bill, sex, smoker, day, time, size):
    # Download and load the pre-trained model
    model_path = hf_hub_download('merve/tips5wx_sbh5-tip-regression', 'sklearn_model.joblib')
    model = joblib.load(model_path)
    # Prepare the data for prediction
    predict_data = [[total_bill, sex, smoker, day, time, size]]
    # Predict the tip
    prediction = model.predict(predict_data)
    return prediction