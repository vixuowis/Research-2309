# function_import --------------------

from huggingface_hub import hf_hub_download
import joblib

# function_code --------------------

def predict_tip(total_bill, sex, smoker, day, time, size):
    """
    Predicts the tip amount based on the given parameters.

    Args:
        total_bill (float): Total bill amount.
        sex (int): Sex of the customer (0 for male, 1 for female).
        smoker (int): Whether the customer is a smoker (0 for no, 1 for yes).
        day (int): Day of the week (0 for Sunday, 1 for Monday, ..., 6 for Saturday).
        time (int): Time of the day (0 for lunch, 1 for dinner).
        size (int): Size of the party.

    Returns:
        float: Predicted tip amount.

    Raises:
        RepositoryNotFoundError: If the model repository is not found.
    """
    model_path = hf_hub_download('merve/tips5wx_sbh5-tip-regression', 'sklearn_model.joblib')
    model = joblib.load(model_path)
    predict_data = [[total_bill, sex, smoker, day, time, size]]
    prediction = model.predict(predict_data)
    return prediction[0]

# test_function_code --------------------

def test_predict_tip():
    """Tests the predict_tip function."""
    assert abs(predict_tip(39.42, 0, 0, 6, 0, 4) - 5.0) < 0.1
    assert abs(predict_tip(20.0, 1, 0, 3, 1, 2) - 3.0) < 0.1
    assert abs(predict_tip(10.0, 0, 1, 4, 0, 1) - 2.0) < 0.1
    return 'All Tests Passed'

# call_test_function_code --------------------

test_predict_tip()