from transformers import pipeline


def predict_pokemon_hp(input_data):
    """
    This function predicts the HP of a Pokemon based on its attributes using a regression model.
    The model is loaded from Hugging Face's model hub.
    
    Args:
    input_data (dict): A dictionary containing the Pokemon's attributes.
    
    Returns:
    float: The predicted HP of the Pokemon.
    """
    # Create a regression model using the pipeline function
    hp_predictor = pipeline('regression', model='julien-c/pokemon-predict-hp')
    
    # Predict the HP of the Pokemon
    predicted_hp = hp_predictor(input_data)[0]['score']
    
    return predicted_hp