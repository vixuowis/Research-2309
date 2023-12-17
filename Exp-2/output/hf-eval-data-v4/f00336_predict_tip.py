# requirements_file --------------------

!pip install -U joblib huggingface_hub

# function_import --------------------

from huggingface_hub import hf_hub_download
import joblib

# function_code --------------------

def predict_tip(total_bill, sex, smoker, day, time, size):
    """
    Predict the tip amount for a customer based on their bill and other attributes.

    Parameters:
        total_bill (float): Total bill amount.
        sex (int): Gender of the customer (0 for Female, 1 for Male).
        smoker (int): Smoking status (0 for No, 1 for Yes).
        day (int): Day of the week coded as 0 for Thurs, 1 for Fri, 2 for Sat, 3 for Sun.
        time (int): Time of day (0 for Lunch, 1 for Dinner).
        size (int): Party size.

    Returns:
        float: The predicted tip amount.
    """
    # Load the pre-trained model
    model_path = hf_hub_download('merve/tips5wx_sbh5-tip-regression', 'sklearn_model.joblib')
    model = joblib.load(model_path)

    # Create the input data array
    predict_data = [[total_bill, sex, smoker, day, time, size]]

    # Predict the tip using the model
    prediction = model.predict(predict_data)
    return prediction[0]

# test_function_code --------------------

def test_predict_tip():
    print("Testing predict_tip function.")

    # Test case 1: Example data input
    tip_prediction = predict_tip(50.81, 1, 0, 2, 1, 3)
    assert tip_prediction > 0, f"Test case failed: prediction = {tip_prediction}"

    # Test case 2: Another set of data
    tip_prediction = predict_tip(15.36, 0, 1, 1, 0, 2)
    assert tip_prediction > 0, f"Test case failed: prediction = {tip_prediction}"

    # Test case 3: Edge case with minimum possible inputs
    tip_prediction = predict_tip(3.07, 0, 0, 0, 0, 1)
    assert tip_prediction > 0, f"Test case failed: prediction = {tip_prediction}"
    print("All tests passed!")

# Execute the test function
test_predict_tip()