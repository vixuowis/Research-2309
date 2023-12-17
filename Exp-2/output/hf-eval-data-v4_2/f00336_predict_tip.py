# requirements_file --------------------

!pip install -U huggingface_hub joblib scikit-learn

# function_import --------------------

from huggingface_hub import hf_hub_download
import joblib

# function_code --------------------

def predict_tip(total_bill, sex, smoker, day, time, size):
    """Predict the tip amount for a customer based on input features.

    Args:
        total_bill (float): The total bill amount of the customer.
        sex (int): Gender of the customer encoded as 0 for male and 1 for female.
        smoker (int): Whether the customer is a smoker encoded as 0 for no and 1 for yes.
        day (int): Day when the customer visited encoded as 0 for Thursday, 1 for Friday, 2 for Saturday, 3 for Sunday.
        time (int): Time of visit encoded as 0 for Dinner and 1 for Lunch.
        size (int): Party size of the customer.

    Returns:
        float: Predicted tip amount.

    Raises:
        ValueError: If any input value is out of expected range or type.
    """
    # Validate input values
    if not isinstance(total_bill, (int, float)):
        raise ValueError("Total bill must be a numerical value.")
    if not 0 <= sex <= 1:
        raise ValueError("Sex must be 0 (male) or 1 (female).")
    if not 0 <= smoker <= 1:
        raise ValueError("Smoker must be 0 (no) or 1 (yes).")
    if not 0 <= day <= 3:
        raise ValueError("Day must be between 0 (Thursday) and 3 (Sunday).")
    if not 0 <= time <= 1:
        raise ValueError("Time must be 0 (Dinner) or 1 (Lunch).")
    if not 1 <= size:
        raise ValueError("Size must be a positive integer.")

    # Load the pre-trained model from the Hugging Face Hub
    model_path = hf_hub_download('merve/tips5wx_sbh5-tip-regression', 'sklearn_model.joblib')
    model = joblib.load(model_path)

    # Predict the tip
    prediction = model.predict([[total_bill, sex, smoker, day, time, size]])[0]
    return prediction

# test_function_code --------------------

def test_predict_tip():
    print("Testing started.")

    # Test case 1: Common case
    print("Testing case [1/3] started.")
    assert predict_tip(50.81, 0, 1, 2, 0, 3) > 0, "Test case [1/3] failed: The predicted tip should be greater than 0."

    # Test case 2: Non-smoker
    print("Testing case [2/3] started.")
    assert predict_tip(34.83, 1, 0, 0, 1, 4) > 0, "Test case [2/3] failed: The predicted tip should be greater than 0."

    # Test case 3: Large party
    print("Testing case [3/3] started.")
    assert predict_tip(48.17, 0, 0, 1, 0, 6) > 0, "Test case [3/3] failed: The predicted tip should be greater than 0."
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_tip()