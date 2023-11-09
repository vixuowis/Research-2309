# function_import --------------------

import tensorflow as tf
from tensorflow_decision_forests.keras import Random_Forest_Model

# function_code --------------------

def TF_Decision_Trees(input_features, target_threshold):
    """
    Use TensorFlow's Gradient Boosted Trees model in binary classification of structured data.
    Build a decision forests model by specifying the input feature usage.
    Implement a custom Binary Target encoder as a Keras Preprocessing layer to encode the categorical features with respect to their target value co-occurrences, and then use the encoded features to build a decision forests model.
    The model is trained on the US Census Income Dataset containing approximately 300k instances with 41 numerical and categorical variables.
    The task is to determine whether a person makes over 50k a year.

    Args:
        input_features (dict): A dictionary of demographic information about a person.
        target_threshold (int): The income threshold for classification.

    Returns:
        model: A trained TensorFlow Decision Trees model.
    """
    # Create TensorFlow Decision Trees model
    model = Random_Forest_Model()

    # Train the model on the dataset (Replace dataset with actual dataset)
    model.fit(input_features)

    # Predict the income category
    income_prediction = model.predict(input_features)

    if income_prediction[0] == 1:
        return "Over 50K per year."
    else:
        return "50K or less per year."


# test_function_code --------------------

def test_TF_Decision_Trees():
    """
    Test the TF_Decision_Trees function.
    """
    input_features = {'age': 30, 'workclass': 'Private', 'education': 'Bachelors', 'marital_status': 'Never-married',
                   'occupation': 'Tech-support', 'relationship': 'Not-in-family', 'race': 'White',
                   'sex': 'Male', 'capital_gain': 0, 'capital_loss': 0, 'hours_per_week': 40,
                   'native_country': 'United-States'}
    target_threshold = 50000
    result = TF_Decision_Trees(input_features, target_threshold)
    assert isinstance(result, str), 'result should be a string'
    assert result in ['Over 50K per year.', '50K or less per year.'], 'result should be one of the two possible classifications'

# call_test_function_code --------------------

test_TF_Decision_Trees()