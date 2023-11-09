import tensorflow as tf
from TF_Decision_Trees import TF_Decision_Trees


def predict_income_category(input_features):
    '''
    Predict the income category of a person based on their demographic information.
    The model uses TensorFlow's Gradient Boosted Trees for binary classification of structured data.
    
    Args:
    input_features (dict): A dictionary containing demographic information of a person.
    
    Returns:
    str: A string indicating the income category of the person.
    '''
    # Create TensorFlow Decision Trees model
    model = TF_Decision_Trees(input_features, target_threshold=50_000)

    # Train the model on the dataset (Replace dataset with actual dataset)
    model.fit(dataset)

    # Predict the income category
    income_prediction = model.predict(input_features)

    if income_prediction[0] == 1:
        return "Over 50K per year."
    else:
        return "50K or less per year."
